from configs import (
    EMBEDDING_MODEL, DEFAULT_VS_TYPE, ZH_TITLE_ENHANCE,
    CHUNK_SIZE, OVERLAP_SIZE,
    logger, log_verbose
)
from server.knowledge_base.utils import (
    get_file_path, list_kbs_from_folder,
    list_files_from_folder, files2docs_in_thread,
    KnowledgeFile
)
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.models.conversation_model import ConversationModel
from server.db.models.message_model import MessageModel
from server.db.repository.knowledge_file_repository import add_file_to_db # ensure Models are imported
from server.db.repository.knowledge_metadata_repository import add_summary_to_db

from server.db.base import Base, engine
from server.db.session import session_scope
import os
from dateutil.parser import parse
from typing import Literal, List


def create_tables():
    """
    创建数据库表

    :return: None
    """
    Base.metadata.create_all(bind=engine)


def reset_tables():
    """
    重置数据库表结构

    :return: None
    """
    Base.metadata.drop_all(bind=engine)
    create_tables()


def import_from_db(
    sqlite_path: str = None,
    # csv_path: str = None,
) -> bool:
    """
    从备份数据库中导入数据到 info.db。
    当 info.db 结构发生变化但无需重新向量化时，适用于版本升级。
    请确保两边数据库表名一致，需要导入的字段名一致。
    当前仅支持 sqlite。
    """
    import sqlite3 as sql
    from pprint import pprint

    models = list(Base.registry.mappers)

    try:
        con = sql.connect(sqlite_path)
        con.row_factory = sql.Row
        cur = con.cursor()
        tables = [x["name"] for x in cur.execute("select name from sqlite_master where type='table'").fetchall()]
        for model in models:
            table = model.local_table.fullname
            if table not in tables:
                continue
            print(f"processing table: {table}")
            with session_scope() as session:
                for row in cur.execute(f"select * from {table}").fetchall():
                    data = {k: row[k] for k in row.keys() if k in model.columns}
                    if "create_time" in data:
                        data["create_time"] = parse(data["create_time"])
                    pprint(data)
                    session.add(model.class_(**data))
        con.close()
        return True
    except Exception as e:
        print(f"无法读取备份数据库：{sqlite_path}。错误信息：{e}")
        return False


def file_to_kbfile(kb_name: str, files: List[str]) -> List[KnowledgeFile]:
    # 创建一个空列表，用于存储转换后的知识文件
    kb_files = []
    # 遍历文件列表
    for file in files:
        try:
            # 创建一个知识文件对象，指定文件名和知识库名称
            kb_file = KnowledgeFile(filename=file, knowledge_base_name=kb_name)
            # 将知识文件对象添加到列表中
            kb_files.append(kb_file)
        except Exception as e:
            # 如果出现异常，记录错误信息
            msg = f"{e}，已跳过"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
    # 返回转换后的知识文件列表
    return kb_files


def folder2db(
    kb_names: List[str],
    mode: Literal["recreate_vs", "update_in_db", "increment"],
    vs_type: Literal["faiss", "milvus", "pg", "chromadb"] = DEFAULT_VS_TYPE,
    embed_model: str = EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = OVERLAP_SIZE,
    zh_title_enhance: bool = ZH_TITLE_ENHANCE,
):
    """
    使用本地文件夹中的文件填充数据库和向量存储。
    设置参数`mode`为：
        recreate_vs：重新创建所有向量存储并使用本地文件夹中的文件填充数据库信息
        fill_info_only（禁用）：不创建向量存储，仅使用本地文件填充数据库信息
        update_in_db：使用仅存在于数据库中的本地文件更新向量存储和数据库信息
        increment：仅创建本地文件中不存在于数据库中的向量存储和数据库信息
    """

    def files2vs(kb_name: str, kb_files: List[KnowledgeFile]):
        for success, result in files2docs_in_thread(kb_files,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    zh_title_enhance=zh_title_enhance):
            if success:
                _, filename, docs = result
                print(f"正在将 {kb_name}/{filename} 添加到向量库，共包含{len(docs)}条文档")
                kb_file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
                kb_file.splited_docs = docs
                kb.add_doc(kb_file=kb_file, not_refresh_vs_cache=True)
            else:
                print(result)

    kb_names = kb_names or list_kbs_from_folder()
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service(kb_name, vs_type, embed_model)
        if not kb.exists():
            kb.create_kb()

        # 清除向量库，从本地文件重建
        if mode == "recreate_vs":
            kb.clear_vs()
            kb.create_kb()
            kb_files = file_to_kbfile(kb_name, list_files_from_folder(kb_name))
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        # # 由于现在数据库存了很多与文本切分相关的信息，单纯存储文件信息意义不大，该功能取消。
        # elif mode == "fill_info_only":
        #     files = list_files_from_folder(kb_name)
        #     kb_files = file_to_kbfile(kb_name, files)
        #     for kb_file in kb_files:
        #         add_file_to_db(kb_file)
        #         print(f"已将 {kb_name}/{kb_file.filename} 添加到数据库")
        # 以数据库中文件列表为基准，利用本地文件更新向量库
        elif mode == "update_in_db":
            files = kb.list_files()
            kb_files = file_to_kbfile(kb_name, files)
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        # 对比本地目录与数据库中的文件列表，进行增量向量化
        elif mode == "increment":
            db_files = kb.list_files()
            folder_files = list_files_from_folder(kb_name)
            files = list(set(folder_files) - set(db_files))
            kb_files = file_to_kbfile(kb_name, files)
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        else:
            print(f"不支持的迁移模式：{mode}")


def prune_db_docs(kb_names: List[str]):
    """
    删除数据库中不存在于本地文件夹中的文档。
    用于在用户删除文件浏览器中的一些文档文件后删除数据库文档。
    """
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is not None:
            files_in_db = kb.list_files()
            files_in_folder = list_files_from_folder(kb_name)
            files = list(set(files_in_db) - set(files_in_folder))
            kb_files = file_to_kbfile(kb_name, files)
            for kb_file in kb_files:
                kb.delete_doc(kb_file, not_refresh_vs_cache=True)
                print(f"成功删除文件：{kb_name}/{kb_file.filename}的文档")
            kb.save_vector_store()


def prune_folder_files(kb_names: List[str]):
    """
    删除本地文件夹中数据库中不存在的文档文件。
    用于通过删除未使用的文档文件来释放本地磁盘空间。
    """
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is not None:
            files_in_db = kb.list_files()
            files_in_folder = list_files_from_folder(kb_name)
            files = list(set(files_in_folder) - set(files_in_db))
            for file in files:
                os.remove(get_file_path(kb_name, file))
                print(f"成功删除文件: {kb_name}/{file}")
