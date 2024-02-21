# 1. 环境设置与导入
# - 检测CPU核心数量并设置NUMEXPR_MAX_THREADS环境变量以优化numexpr库的性能。
# - 将当前文件目录的上级目录添加到sys.path中，以便导入项目模块。

# 2. 配置导入
# - 从configs模块导入配置项，如日志路径、日志详细度、日志记录器、模型列表、嵌入模型、文本拆分器名称等。
# - 从server.utils模块导入各种工具函数和类，如FastAPI框架、模型工作者配置获取等。

# 3. 数据库表创建
# - 使用create_tables函数来初始化数据库表结构，为存储知识库数据提供支持。

# 4. FastAPI应用创建
# - 提供create_controller_app、create_model_worker_app和create_openai_api_app等函数，用于创建不同服务的FastAPI应用程序。
#   这些应用程序分别对应控制器、模型工作者和OpenAI API服务器。
# - 实现了模型的动态加载与释放，以及模型工作者之间的调度。

# 5. 服务器运行逻辑
# - run_controller、run_model_worker、run_openai_api、run_api_server和run_webui等函数，分别用于启动控制器、模型工作者、
#   OpenAI API服务器、API服务器和WebUI界面。
# - 使用子进程和事件通知机制来异步启动和管理这些服务。

# 6. 命令行参数解析
# - 使用argparse库解析命令行参数，以支持不同的启动模式和配置。

# 7. 服务启动与管理
# - 在主函数中根据命令行参数选择性地启动各种服务，并通过队列和事件来协调服务之间的通信和状态管理。
# - 提供了优雅的退出机制和错误处理，确保服务能够安全地停止。

# 8. 日志和配置信息的打印
# - 在服务启动前后打印出详细的配置信息和服务器运行状态，便于调试和监控。


import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import Process
from datetime import datetime
from pprint import pprint
from langchain_core._api import deprecated

try:
    import numexpr

    # 检测CPU核心数量
    n_cores = numexpr.utils.detect_number_of_cores()
    # 将核心数量设置为NUMEXPR_MAX_THREADS的环境变量
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

# 将当前文件所在目录的上级目录添加到sys.path中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 从configs模块中导入以下配置项
from configs import (
    LOG_PATH,  # 日志文件路径
    log_verbose,  # 是否详细输出日志
    logger,  # 日志记录器
    LLM_MODELS,  # 语言模型列表
    EMBEDDING_MODEL,  # 嵌入模型
    TEXT_SPLITTER_NAME,  # 文本拆分器名称
    FSCHAT_CONTROLLER,  # FSChat控制器地址
    FSCHAT_OPENAI_API,  # FSChat OpenAI API地址
    FSCHAT_MODEL_WORKERS,  # FSChat模型工作者地址
    API_SERVER,  # API服务器地址
    WEBUI_SERVER,  # WebUI服务器地址
    HTTPX_DEFAULT_TIMEOUT,  # HTTPX默认超时时间
)
# 从server.utils模块中导入以下函数
from server.utils import (
    fschat_controller_address,  # FSChat控制器地址
    fschat_model_worker_address,  # FSChat模型工作者地址
    fschat_openai_api_address,  # FSChat OpenAI API地址
    get_httpx_client,  # 获取HTTPX客户端
    get_model_worker_config,  # 获取模型工作者配置
    MakeFastAPIOffline,  # 快速创建离线FastAPI应用程序
    FastAPI,  # FastAPI框架
    llm_device,  # LLM设备
    embedding_device,  # 嵌入设备
)

# 从server.knowledge_base.migrate模块中导入以下函数
from server.knowledge_base.migrate import create_tables
# 导入argparse模块
import argparse
# 从typing模块中导入List和Dict类型
from typing import List, Dict
# 导入VERSION变量
from configs import VERSION


# deprecated该装饰器用于标记一个函数或方法为已废弃，并提供废弃的时间、废弃原因和废弃版本信息。
# since：废弃的时间，即从哪个版本开始废弃。
# message：废弃的原因，即为什么需要废弃。
# removal：废弃的版本，即在哪个版本中将完全移除该功能。
@deprecated(
    since="0.3.0",
    message="模型启动功能将于 Langchain-Chatchat 0.3.x重写,支持更多模式和加速启动，0.2.x中相关功能将废弃",
    removal="0.3.0")
def create_controller_app(
    dispatch_method: str,
    log_level: str = "INFO",
) -> FastAPI:
    """
    创建一个FastAPI应用程序的控制器。

    参数：
    dispatch_method (str): 控制器的调度方法。
    log_level (str, optional): 日志级别，默认为"INFO"。

    返回：
    FastAPI: 创建的FastAPI应用程序。

    """
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import app, Controller, logger
    logger.setLevel(log_level)

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller

    MakeFastAPIOffline(app)
    app.title = "FastChat Controller"
    app._controller = controller
    return app


def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    """
    创建一个模型工作者应用程序。

    参数:
        log_level (str): 日志级别，默认为"INFO"。
        **kwargs: 其他关键字参数。

    返回:
        FastAPI: 模型工作者应用程序。

    """

    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH

    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    for k, v in kwargs.items():
        setattr(args, k, v)

    if worker_class := kwargs.get("langchain_model"):  # Langchian支持的模型不用做操作
        from fastchat.serve.base_model_worker import app
        worker = ""
    # 在线模型API
    elif worker_class := kwargs.get("worker_class"):
        from fastchat.serve.base_model_worker import app

        worker = worker_class(model_names=args.model_names,
                              controller_addr=args.controller_address,
                              worker_addr=args.worker_address)
        sys.modules["fastchat.serve.base_model_worker"].logger.setLevel(log_level)
    # 本地模型
    else:
        from configs.model_config import VLLM_MODEL_DICT
        if kwargs["model_names"][0] in VLLM_MODEL_DICT and args.infer_turbo == "vllm":
            import fastchat.serve.vllm_worker
            from fastchat.serve.vllm_worker import VLLMWorker, app, worker_id
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs

            # 设置分词器为模型路径
            args.tokenizer = args.model_path
            # 设置分词器模式为自动
            args.tokenizer_mode = 'auto'
            # 设置信任远程代码为True
            args.trust_remote_code = True
            # 设置下载目录为None
            args.download_dir = None
            # 设置加载格式为自动
            args.load_format = 'auto'
            # 设置数据类型为自动
            args.dtype = 'auto'
            # 设置随机种子为0
            args.seed = 0
            # 设置使用Ray进行工作进程
            args.worker_use_ray = False
            # 设置管道并行大小为1
            args.pipeline_parallel_size = 1
            # 设置张量并行大小为1
            args.tensor_parallel_size = 1
            # 设置块大小为16
            args.block_size = 16
            # 设置交换空间为4 GiB
            args.swap_space = 4  # GiB
            # 设置GPU内存利用率阈值为0.90
            args.gpu_memory_utilization = 0.90
            # 一个批次中的最大令牌（tokens）数量，这个取决于你的显卡和大模型设置，设置太大显存会不够
            args.max_num_batched_tokens = None  
            # 设置最大序列数为256
            args.max_num_seqs = 256
            # 禁用日志统计信息
            args.disable_log_stats = False
            # 设置卷积模板为None
            args.conv_template = None
            # 限制每个工作进程的最大并发数为5
            args.limit_worker_concurrency = 5
            # 不进行注册
            args.no_register = False
            # vllm worker的切分是tensor并行，这里填写显卡的数量
            args.num_gpus = 1
            # 设置args对象的engine_use_ray属性为False
            args.engine_use_ray = False
            # 设置args对象的disable_log_requests属性为False
            args.disable_log_requests = False

            # 0.2.1 vllm后要加的参数, 但是这里不需要
            # 设置最大模型长度为None
            args.max_model_len = None
            # 设置修订版本为None
            args.revision = None
            # 设置量化为None
            args.quantization = None
            # 设置最大日志长度为None
            args.max_log_len = None
            # 设置分词器修订版本为None
            args.tokenizer_revision = None

            # 0.2.2 vllm需要新加的参数
            args.max_paddings = 256

            # 如果命令行参数中指定了模型路径，则将该路径赋值给args的model属性
            if args.model_path:
                args.model = args.model_path
            # 如果命令行参数中指定的GPU数量大于1，则将该数量赋值给args的tensor_parallel_size属性
            if args.num_gpus > 1:
                args.tensor_parallel_size = args.num_gpus

            # 将命令行参数中的其他参数赋值给args对象
            for k, v in kwargs.items():
                setattr(args, k, v)

            # 使用命令行参数创建AsyncEngineArgs对象，并使用该对象创建AsyncLLMEngine对象
            engine_args = AsyncEngineArgs.from_cli_args(args)
            engine = AsyncLLMEngine.from_engine_args(engine_args)

            # 创建VLLMWorker对象
            worker = VLLMWorker(
                # 控制器地址
                controller_addr=args.controller_address,
                # 工作器地址
                worker_addr=args.worker_address,
                # 工作器ID
                worker_id=worker_id,
                # 模型路径
                model_path=args.model_path,
                # 模型名称列表
                model_names=args.model_names,
                # 限制工作器并发数
                limit_worker_concurrency=args.limit_worker_concurrency,
                # 不进行注册
                no_register=args.no_register,
                # LLM引擎
                llm_engine=engine,
                # 转换模板
                conv_template=args.conv_template,
            )
            # 将engine对象赋值给sys.modules["fastchat.serve.vllm_worker"].engine
            sys.modules["fastchat.serve.vllm_worker"].engine = engine
            # 将worker对象赋值给sys.modules["fastchat.serve.vllm_worker"].worker
            sys.modules["fastchat.serve.vllm_worker"].worker = worker
            # 设置sys.modules["fastchat.serve.vllm_worker"].logger的日志级别为log_level
            sys.modules["fastchat.serve.vllm_worker"].logger.setLevel(log_level)

        else:
            from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker, worker_id

            # GPU的编号,如果有多个GPU，可以设置为"0,1,2,3"
            args.gpus = "0"
            # 设置最大GPU内存为22GiB
            args.max_gpu_memory = "22GiB"
            # model worker的切分是model并行，这里填写显卡的数量
            args.num_gpus = 1 

            # 是否加载8位模型
            args.load_8bit = False
            # CPU offloading策略
            args.cpu_offloading = None
            # GPTQ检查点路径
            args.gptq_ckpt = None
            # GPTQ的wbits大小
            args.gptq_wbits = 16
            # GPTQ的groupsize大小
            args.gptq_groupsize = -1
            # 是否按顺序执行GPTQ
            args.gptq_act_order = False
            # AWQ检查点路径
            args.awq_ckpt = None
            # AWQ的wbits大小
            args.awq_wbits = 16
            # AWQ的groupsize大小
            args.awq_groupsize = -1
            # 模型名称列表
            args.model_names = [""]
            # 卷积模板路径
            args.conv_template = None
            # 工作进程并发数限制
            args.limit_worker_concurrency = 5
            # 数据流间隔时间
            args.stream_interval = 2
            # 是否注册模型
            args.no_register = False
            # 是否在截断时嵌入
            args.embed_in_truncate = False
            # 遍历kwargs字典中的键值对
            for k, v in kwargs.items():
                # 为args对象设置属性
                setattr(args, k, v)

            # 如果args对象中存在gpus属性
            if args.gpus:
                # 如果args对象中num_gpus属性为None
                if args.num_gpus is None:
                    # 将gpus属性按逗号分隔后的元素个数赋值给num_gpus属性
                    args.num_gpus = len(args.gpus.split(','))
                # 如果gpus属性按逗号分隔后的元素个数小于num_gpus属性的值
                if len(args.gpus.split(",")) < args.num_gpus:
                    # 抛出ValueError异常
                    raise ValueError(
                        f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                    )
                # 将gpus属性的值设置为CUDA_VISIBLE_DEVICES环境变量的值
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

            # 创建GptqConfig对象
            gptq_config = GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,  # 指定GPT-Q的检查点路径，如果不存在则使用args.model_path指定的模型路径
                wbits=args.gptq_wbits,  # 设置GPT-Q的位宽
                groupsize=args.gptq_groupsize,  # 设置GPT-Q的分组大小
                act_order=args.gptq_act_order,  # 设置GPT-Q的激活函数顺序
            )

            # 创建AWQConfig对象
            awq_config = AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,  # 指定AWQConfig对象的checkpoint路径，如果awq_ckpt参数存在则使用awq_ckpt参数，否则使用model_path参数
                wbits=args.awq_wbits,  # 指定AWQConfig对象的量化字节数
                groupsize=args.awq_groupsize,  # 指定AWQConfig对象的分组大小
            )

            worker = ModelWorker(
                controller_addr=args.controller_address,  # 控制器地址
                worker_addr=args.worker_address,  # 工作器地址
                worker_id=worker_id,  # 工作器ID
                model_path=args.model_path,  # 模型路径
                model_names=args.model_names,  # 模型名称
                limit_worker_concurrency=args.limit_worker_concurrency,  # 限制工作器并发
                no_register=args.no_register,  # 是否不注册
                device=args.device,  # 设备
                num_gpus=args.num_gpus,  # GPU数量
                max_gpu_memory=args.max_gpu_memory,  # 最大GPU内存
                load_8bit=args.load_8bit,  # 是否加载8位模型
                cpu_offloading=args.cpu_offloading,  # 是否启用CPU卸载
                gptq_config=gptq_config,  # GPTQ配置
                awq_config=awq_config,  # AWQ配置
                stream_interval=args.stream_interval,  # 数据流间隔
                conv_template=args.conv_template,  # 卷积模板
                embed_in_truncate=args.embed_in_truncate,  # 是否在截断中嵌入
            )
            sys.modules["fastchat.serve.model_worker"].args = args
            sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
            sys.modules["fastchat.serve.model_worker"].worker = worker
            sys.modules["fastchat.serve.model_worker"].logger.setLevel(log_level)

    MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    app._worker = worker
    return app


def create_openai_api_app(
    controller_address: str,  # 控制器地址
    api_keys: List = [],  # API密钥列表，默认为空列表
    log_level: str = "INFO",  # 日志级别，默认为INFO
) -> FastAPI:  # 返回一个FastAPI对象
    import fastchat.constants  # 导入fastchat.constants模块
    fastchat.constants.LOGDIR = LOG_PATH  # 设置日志目录为LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings  # 导入app、CORSMiddleware和app_settings
    from fastchat.utils import build_logger  # 导入build_logger函数
    logger = build_logger("openai_api", "openai_api.log")  # 创建一个名为"openai_api"的日志记录器，并将日志输出到"openai_api.log"文件中
    logger.setLevel(log_level)  # 设置日志记录器的日志级别为log_level

    app.add_middleware(
        CORSMiddleware,  # 添加中间件
        allow_credentials=True,  # 允许携带凭证
        allow_origins=["*"],  # 允许所有来源
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头部信息
    )

    sys.modules["fastchat.serve.openai_api_server"].logger = logger  # 将logger赋值给fastchat.serve.openai_api_server模块的logger属性
    app_settings.controller_address = controller_address  # 设置app_settings的controller_address属性为controller_address
    app_settings.api_keys = api_keys  # 设置app_settings的api_keys属性为api_keys

    MakeFastAPIOffline(app)  # 调用MakeFastAPIOffline函数，将app作为参数传入
    app.title = "FastChat OpeanAI API Server"  # 设置app的title属性为"FastChat OpeanAI API Server"
    return app  # 返回app对象


def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    """
    设置应用程序的事件处理函数。

    Args:
        app (FastAPI): FastAPI应用程序对象。
        started_event (mp.Event, optional): 用于指示应用程序是否已启动的事件对象。默认为None。

    Returns:
        None
    """
    @app.on_event("startup")
    async def on_startup():
        """
        在应用程序启动时触发的事件处理函数。

        Args:
            None

        Returns:
            None
        """
        if started_event is not None:
            started_event.set()


def run_controller(log_level: str = "INFO", started_event: mp.Event = None):
    """
    运行控制器

    参数:
    log_level (str): 日志级别，默认为"INFO"
    started_event (mp.Event): 开始事件，默认为None

    返回:
    无
    """

    import uvicorn
    import httpx
    from fastapi import Body
    import time
    import sys
    from server.utils import set_httpx_config
    set_httpx_config()

    app = create_controller_app(
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),
        log_level=log_level,
    )
    _set_app_event(app, started_event)

    # 添加释放和加载模型工人的接口
    @app.post("/release_worker")
    def release_worker(
            model_name: str = Body(..., description="要释放模型的名称", samples=["chatglm-6b"]),
            # worker_address: str = Body(None, description="要释放模型的地址，与名称二选一", samples=[FSCHAT_CONTROLLER_address()]),
            new_model_name: str = Body(None, description="释放后加载该模型"),
            keep_origin: bool = Body(False, description="不释放原模型，加载新模型")
    ) -> Dict:
        """
        释放模型工人

        参数:
        model_name (str): 要释放的模型名称
        new_model_name (str): 释放后加载的模型名称
        keep_origin (bool): 是否保留原模型，加载新模型

        返回:
        Dict: 响应结果
        """
        available_models = app._controller.list_models()
        if new_model_name in available_models:
            msg = f"要切换的LLM模型 {new_model_name} 已经存在"
            logger.info(msg)
            return {"code": 500, "msg": msg}

        if new_model_name:
            logger.info(f"开始切换LLM模型：从 {model_name} 到 {new_model_name}")
        else:
            logger.info(f"即将停止LLM模型： {model_name}")

        if model_name not in available_models:
            msg = f"the model {model_name} is not available"
            logger.error(msg)
            return {"code": 500, "msg": msg}

        worker_address = app._controller.get_worker_address(model_name)
        if not worker_address:
            msg = f"can not find model_worker address for {model_name}"
            logger.error(msg)
            return {"code": 500, "msg": msg}

        with get_httpx_client() as client:
            r = client.post(worker_address + "/release",
                            json={"new_model_name": new_model_name, "keep_origin": keep_origin})
            if r.status_code != 200:
                msg = f"failed to release model: {model_name}"
                logger.error(msg)
                return {"code": 500, "msg": msg}

        if new_model_name:
            timer = HTTPX_DEFAULT_TIMEOUT  # 等待新模型工人注册
            while timer > 0:
                models = app._controller.list_models()
                if new_model_name in models:
                    break
                time.sleep(1)
                timer -= 1
            if timer > 0:
                msg = f"成功切换模型：从 {model_name} 到 {new_model_name}"
                logger.info(msg)
                return {"code": 200, "msg": msg}
            else:
                msg = f"切换模型失败：从 {model_name} 到 {new_model_name}"
                logger.error(msg)
                return {"code": 500, "msg": msg}
        else:
            msg = f"成功释放模型：{model_name}"
            logger.info(msg)
            return {"code": 200, "msg": msg}

    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]

    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def run_model_worker(
    model_name: str = LLM_MODELS[0],  # 模型名称，默认为LLM_MODELS列表的第一个元素
    controller_address: str = "",  # 控制器地址，默认为空字符串
    log_level: str = "INFO",  # 日志级别，默认为"INFO"
    q: mp.Queue = None,  # 用于传递消息的队列，默认为None
    started_event: mp.Event = None,  # 用于通知进程是否已启动的事件，默认为None
):
    import uvicorn  # 引入uvicorn模块
    from fastapi import Body  # 引入Body类
    import sys  # 引入sys模块
    from server.utils import set_httpx_config  # 引入set_httpx_config函数
    set_httpx_config()  # 设置httpx配置

    kwargs = get_model_worker_config(model_name)  # 获取模型工作器的配置参数
    host = kwargs.pop("host")  # 弹出并获取配置参数中的主机地址
    port = kwargs.pop("port")  # 弹出并获取配置参数中的端口号
    kwargs["model_names"] = [model_name]  # 将模型名称添加到配置参数中的模型名称列表中
    kwargs["controller_address"] = controller_address or fschat_controller_address()  # 如果控制器地址为空，则使用默认的控制器地址
    kwargs["worker_address"] = fschat_model_worker_address(model_name)  # 获取模型工作器的地址
    model_path = kwargs.get("model_path", "")  # 获取配置参数中的模型路径，如果不存在则为空字符串
    kwargs["model_path"] = model_path  # 将模型路径添加到配置参数中的模型路径中

    app = create_model_worker_app(log_level=log_level, **kwargs)  # 创建模型工作器的FastAPI应用程序
    _set_app_event(app, started_event)  # 设置应用程序的事件
    if log_level == "ERROR":  # 如果日志级别为"ERROR"
        sys.stdout = sys.__stdout__  # 将标准输出重置为原始的标准输出
        sys.stderr = sys.__stderr__  # 将标准错误重置为原始的标准错误

    # 添加释放和加载模型的接口
    @app.post("/release")  # 添加一个POST请求路由
    def release_model(
        new_model_name: str = Body(None, description="释放后加载该模型"),  # 请求体参数，表示要释放的模型名称，默认为None
        keep_origin: bool = Body(False, description="不释放原模型，加载新模型")  # 请求体参数，表示是否保留原模型，默认为False
    ) -> Dict:  # 返回一个字典类型
        if keep_origin:  # 如果保留原模型
            if new_model_name:  # 如果有新的模型名称
                q.put([model_name, "start", new_model_name])  # 将模型名称、操作类型（"start"）和新的模型名称放入队列中
        else:  # 如果不保留原模型
            if new_model_name:  # 如果有新的模型名称
                q.put([model_name, "replace", new_model_name])  # 将模型名称、操作类型（"replace"）和新的模型名称放入队列中
            else:  # 如果没有新的模型名称
                q.put([model_name, "stop", None])  # 将模型名称、操作类型（"stop"）和None放入队列中
        return {"code": 200, "msg": "done"}  # 返回一个包含状态码和消息的字典

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())  # 启动uvicorn服务器，运行模型工作器应用程序


def run_openai_api(log_level: str = "INFO", started_event: mp.Event = None):
    """
    运行OpenAI API服务器

    参数:
        log_level (str): 日志级别，默认为"INFO"
        started_event (mp.Event): 用于通知进程是否已启动的事件，默认为None
    """
    import uvicorn  # 导入uvicorn模块
    import sys  # 导入sys模块
    from server.utils import set_httpx_config  # 导入set_httpx_config函数
    set_httpx_config()  # 调用set_httpx_config函数

    controller_addr = fschat_controller_address()  # 调用fschat_controller_address函数获取controller_addr
    app = create_openai_api_app(controller_addr, log_level=log_level)  # 调用create_openai_api_app函数创建app对象
    _set_app_event(app, started_event)  # 调用_set_app_event函数设置app事件

    host = FSCHAT_OPENAI_API["host"]  # 获取FSCHAT_OPENAI_API字典中的host键值
    port = FSCHAT_OPENAI_API["port"]  # 获取FSCHAT_OPENAI_API字典中的port键值
    if log_level == "ERROR":  # 如果log_level等于"ERROR"
        sys.stdout = sys.__stdout__  # 将sys.stdout重置为sys.__stdout__
        sys.stderr = sys.__stderr__  # 将sys.stderr重置为sys.__stderr__
    uvicorn.run(app, host=host, port=port)  # 运行uvicorn，传入app对象和host、port参数


def run_api_server(started_event: mp.Event = None, run_mode: str = None):
    """
    运行API服务器

    Args:
        started_event (mp.Event, optional): 事件对象，用于通知API服务器已启动。默认为None。
        run_mode (str, optional): 运行模式。默认为None。
    """
    # 导入必要的模块和函数
    from server.api import create_app
    import uvicorn
    from server.utils import set_httpx_config

    # 设置HTTPX配置
    set_httpx_config()

    # 创建Flask应用程序
    app = create_app(run_mode=run_mode)

    # 设置应用程序事件
    _set_app_event(app, started_event)

    # 获取API服务器的主机和端口
    host = API_SERVER["host"]
    port = API_SERVER["port"]

    # 运行Uvicorn服务器
    uvicorn.run(app, host=host, port=port)


def run_webui(started_event: mp.Event = None, run_mode: str = None):
    """
    运行WebUI界面

    参数:
    started_event (mp.Event): 一个事件对象，用于通知WebUI界面已经启动
    run_mode (str): 运行模式，可选值为"lite"

    返回:
    无
    """

    from server.utils import set_httpx_config
    set_httpx_config()

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]

    cmd = ["streamlit", "run", "webui.py",
           "--server.address", host,
           "--server.port", str(port),
           "--theme.base", "light",
           "--theme.primaryColor", "#165dff",
           "--theme.secondaryBackgroundColor", "#f5f5f5",
           "--theme.textColor", "#000000",
           ]
    if run_mode == "lite":
        cmd += [
            "--",
            "lite",
        ]
    p = subprocess.Popen(cmd)
    started_event.set()
    p.wait()


import argparse

def parse_args() -> argparse.ArgumentParser:
    """
    解析命令行参数并返回解析结果和解析器对象。

    Returns:
        args: 解析结果
        parser: 解析器对象
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--all-webui",
        action="store_true",
        help="运行 fastchat 的 controller/openai_api/model_worker 服务器，运行 api.py 和 webui.py",
        dest="all_webui",
    )
    parser.add_argument(
        "--all-api",
        action="store_true",
        help="运行 fastchat 的 controller/openai_api/model_worker 服务器，运行 api.py",
        dest="all_api",
    )
    parser.add_argument(
        "--llm-api",
        action="store_true",
        help="运行 fastchat 的 controller/openai_api/model_worker 服务器",
        dest="llm_api",
    )
    parser.add_argument(
        "-o",
        "--openai-api",
        action="store_true",
        help="运行 fastchat 的 controller/openai_api 服务器",
        dest="openai_api",
    )
    parser.add_argument(
        "-m",
        "--model-worker",
        action="store_true",
        help="运行 fastchat 的 model_worker 服务器，指定模型名称。如果使用默认的 LLM_MODELS，请指定 --model-name",
        dest="model_worker",
    )
    parser.add_argument(
        "-n",
        "--model-name",
        type=str,
        nargs="+",
        default=LLM_MODELS,
        help="指定 model_worker 服务器的模型名称。使用空格分隔以启动多个模型工作者。",
        dest="model_name",
    )
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        help="指定工作者注册的控制器地址，默认为 FSCHAT_CONTROLLER",
        dest="controller_address",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="运行 api.py 服务器",
        dest="api",
    )
    parser.add_argument(
        "-p",
        "--api-worker",
        action="store_true",
        help="运行在线模型 API，如 zhipuai",
        dest="api_worker",
    )
    parser.add_argument(
        "-w",
        "--webui",
        action="store_true",
        help="运行 webui.py 服务器",
        dest="webui",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="减少 fastchat 服务的日志信息",
        dest="quiet",
    )
    parser.add_argument(
        "-i",
        "--lite",
        action="store_true",
        help="以 Lite 模式运行：仅支持在线 API 的 LLM 对话、搜索引擎对话",
        dest="lite",
    )
    args = parser.parse_args()
    return args, parser


def dump_server_info(after_start=False, args=None):
    """
    打印服务器配置信息

    参数:
    after_start (bool): 是否在启动后打印
    args (dict): 参数字典

    返回:
    无
    """
    import platform
    import langchain
    import fastchat
    from server.utils import api_address, webui_address

    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print(f"操作系统：{platform.platform()}.")
    print(f"python版本：{sys.version}")
    print(f"项目版本：{VERSION}")
    print(f"langchain版本：{langchain.__version__}. fastchat版本：{fastchat.__version__}")
    print("\n")

    models = LLM_MODELS
    if args and args.model_name:
        models = args.model_name

    print(f"当前使用的分词器：{TEXT_SPLITTER_NAME}")
    print(f"当前启动的LLM模型：{models} @ {llm_device()}")

    for model in models:
        pprint(get_model_worker_config(model))
    print(f"当前Embbedings模型： {EMBEDDING_MODEL} @ {embedding_device()}")

    if after_start:
        print("\n")
        print(f"服务端运行信息：")
        if args.openai_api:
            print(f"    OpenAI API Server: {fschat_openai_api_address()}")
        if args.api:
            print(f"    Chatchat  API  Server: {api_address()}")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")


async def start_main_server():
    import time
    import signal

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """
        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")
        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()
    run_mode = None

    queue = manager.Queue()
    args, parser = parse_args()

    if args.all_webui:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = True

    elif args.all_api:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = False

    elif args.llm_api:
        args.openai_api = True
        args.model_worker = True
        args.api_worker = True
        args.api = False
        args.webui = False

    if args.lite:
        args.model_worker = False
        run_mode = "lite"

    dump_server_info(args=args)

    if len(sys.argv) > 1:
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {LOG_PATH}")

    processes = {"online_api": {}, "model_worker": {}}

    def process_count():
        return len(processes) + len(processes["online_api"]) + len(processes["model_worker"]) - 2

    if args.quiet or not log_verbose:
        log_level = "ERROR"
    else:
        log_level = "INFO"

    controller_started = manager.Event()
    if args.openai_api:
        process = Process(
            target=run_controller,
            name=f"controller",
            kwargs=dict(log_level=log_level, started_event=controller_started),
            daemon=True,
        )
        processes["controller"] = process

        process = Process(
            target=run_openai_api,
            name=f"openai_api",
            daemon=True,
        )
        processes["openai_api"] = process

    model_worker_started = []
    if args.model_worker:
        for model_name in args.model_name:
            config = get_model_worker_config(model_name)
            if not config.get("online_api"):
                e = manager.Event()
                model_worker_started.append(e)
                process = Process(
                    target=run_model_worker,
                    name=f"model_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                processes["model_worker"][model_name] = process

    if args.api_worker:
        for model_name in args.model_name:
            config = get_model_worker_config(model_name)
            if (config.get("online_api")
                    and config.get("worker_class")
                    and model_name in FSCHAT_MODEL_WORKERS):
                e = manager.Event()
                model_worker_started.append(e)
                process = Process(
                    target=run_model_worker,
                    name=f"api_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                processes["online_api"][model_name] = process

    api_started = manager.Event()
    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(started_event=api_started, run_mode=run_mode),
            daemon=True,
        )
        processes["api"] = process

    webui_started = manager.Event()
    if args.webui:
        process = Process(
            target=run_webui,
            name=f"WEBUI Server",
            kwargs=dict(started_event=webui_started, run_mode=run_mode),
            daemon=True,
        )
        processes["webui"] = process

    if process_count() == 0:
        parser.print_help()
    else:
        try:
            # 保证任务收到SIGINT后，能够正常退出
            if p := processes.get("controller"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                controller_started.wait()  # 等待controller启动完成

            if p := processes.get("openai_api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("model_worker", {}).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("online_api", []).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for e in model_worker_started:
                e.wait()

            if p := processes.get("api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                api_started.wait()

            if p := processes.get("webui"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                webui_started.wait()

            dump_server_info(after_start=True, args=args)

            while True:
                cmd = queue.get()
                e = manager.Event()
                if isinstance(cmd, list):
                    model_name, cmd, new_model_name = cmd
                    if cmd == "start":  # 运行新模型
                        logger.info(f"准备启动新模型进程：{new_model_name}")
                        process = Process(
                            target=run_model_worker,
                            name=f"model_worker - {new_model_name}",
                            kwargs=dict(model_name=new_model_name,
                                        controller_address=args.controller_address,
                                        log_level=log_level,
                                        q=queue,
                                        started_event=e),
                            daemon=True,
                        )
                        process.start()
                        process.name = f"{process.name} ({process.pid})"
                        processes["model_worker"][new_model_name] = process
                        e.wait()
                        logger.info(f"成功启动新模型进程：{new_model_name}")
                    elif cmd == "stop":
                        if process := processes["model_worker"].get(model_name):
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            logger.info(f"停止模型进程：{model_name}")
                        else:
                            logger.error(f"未找到模型进程：{model_name}")
                    elif cmd == "replace":
                        if process := processes["model_worker"].pop(model_name, None):
                            logger.info(f"停止模型进程：{model_name}")
                            start_time = datetime.now()
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            process = Process(
                                target=run_model_worker,
                                name=f"model_worker - {new_model_name}",
                                kwargs=dict(model_name=new_model_name,
                                            controller_address=args.controller_address,
                                            log_level=log_level,
                                            q=queue,
                                            started_event=e),
                                daemon=True,
                            )
                            process.start()
                            process.name = f"{process.name} ({process.pid})"
                            processes["model_worker"][new_model_name] = process
                            e.wait()
                            timing = datetime.now() - start_time
                            logger.info(f"成功启动新模型进程：{new_model_name}。用时：{timing}。")
                        else:
                            logger.error(f"未找到模型进程：{model_name}")
                else:
                    logger.error(f"未知命令：{cmd}")

            # for process in processes.get("model_worker", {}).values():
            #     process.join()
            # for process in processes.get("online_api", {}).values():
            #     process.join()

            # for name, process in processes.items():
            #     if name not in ["model_worker", "online_api"]:
            #         if isinstance(p, dict):
            #             for work_process in p.values():
            #                 work_process.join()
            #         else:
            #             process.join()
        except Exception as e:
            logger.error(e)
            logger.warning("Caught KeyboardInterrupt! Setting stop event...")
        finally:
            for p in processes.values():
                logger.warning("Sending SIGKILL to %s", p)
                # Queues and other inter-process communication primitives can break when
                # process is killed, but we don't care here

                if isinstance(p, dict):
                    for process in p.values():
                        process.kill()
                else:
                    p.kill()

            for p in processes.values():
                logger.info("Process status: %s", p)


if __name__ == "__main__":
    # 如果脚本是直接运行的，执行以下代码
    create_tables()  # 创建数据库表

    if sys.version_info < (3, 10):
        # 如果Python版本小于3.10，使用asyncio的默认事件循环
        loop = asyncio.get_event_loop()
    else:
        try:
            # 如果Python版本大于等于3.10，尝试获取正在运行的事件循环
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 如果获取失败，创建一个新的事件循环
            loop = asyncio.new_event_loop()

        # 设置当前事件循环为获取到的事件循环
        asyncio.set_event_loop(loop)

    # 运行事件循环，直到服务器启动完成
    loop.run_until_complete(start_main_server())

# 服务启动后接口调用示例：
# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://localhost:8888/v1"

# model = "chatglm3-6b"

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)
