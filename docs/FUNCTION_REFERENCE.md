# SaddleLLM 函数与方法参考

> 本文档由 `python tools/generate_function_reference.py` 从一方源码静态生成。
> “直接调用”来自 AST 静态扫描，只表示源码中可见的调用，不等同于完整运行时调用图；反射、回调和第三方框架注入不会被完全捕获。

## 覆盖范围

- Python：`saddlellm/` 与兼容包 `saddle_llm/`，共 **2009** 个模块函数、方法和命名的嵌套函数。
- 前端：`spatial-studio/src/`（排除测试），共 **50** 个命名函数、组件、Hook 回调或构造器。
- 涉及 **138** 个含可调用项的源码文件；排除测试、第三方研究仓库、构建产物、依赖目录和匿名内联回调。
- 职责说明优先采用源码 docstring；没有 docstring 时，根据函数名、所属类和函数类型生成明确的用途说明。

## 阅读方式

- 链接会跳到对应源码行。
- `function` 是模块级函数，`method` 是类方法，`nested function` 是函数内部具名回调/辅助函数。
- 调用链和模块边界请先阅读 [ARCHITECTURE.md](ARCHITECTURE.md)。

## `saddle_llm/__init__.py`

共 2 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`__getattr__`](../saddle_llm/__init__.py#L16)<br><sub>`__getattr__(name: str)`</sub> | function | 按需解析未直接绑定的属性，主要用于延迟导入或兼容转发。 | `getattr` |
| [`__dir__`](../saddle_llm/__init__.py#L20)<br><sub>`__dir__()`</sub> | function | 返回该模块或兼容命名空间可发现的公开名称。 | `sorted`, `set`, `globals`, `dir` |

## `saddlellm/AdvancedTechniques.py`

共 33 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`EMA.__init__`](../saddlellm/AdvancedTechniques.py#L60)<br><sub>`__init__(self, model: nn.Module, decay: float=0.999, device=None)`</sub> | method | 初始化 `EMA` 实例及其运行依赖。 | `model.named_parameters`, `detach`, `param.data.clone` |
| [`EMA.update`](../saddlellm/AdvancedTechniques.py#L73)<br><sub>`update(self)`</sub> | method | 更新 EMA shadow 参数。 | `min`, `named_parameters`, `self._get_model`, `add_`, `mul_` |
| [`EMA.apply`](../saddlellm/AdvancedTechniques.py#L83)<br><sub>`apply(self)`</sub> | method | 用 EMA 参数替换模型参数 (用于验证/保存)。 | `self._get_model`, `model.named_parameters`, `param.data.clone`, `param.data.copy_` |
| [`EMA.restore`](../saddlellm/AdvancedTechniques.py#L91)<br><sub>`restore(self)`</sub> | method | 恢复原始参数。 | `self._get_model`, `model.named_parameters`, `param.data.copy_`, `self._backup.clear` |
| [`EMA.apply_and_save`](../saddlellm/AdvancedTechniques.py#L99)<br><sub>`apply_and_save(self, path: str, tokenizer=None)`</sub> | method | 应用 EMA 参数并保存。 | `self.apply`, `save_pretrained`, `self._get_model`, `tokenizer.save_pretrained`, `self.restore`, `logger.info` |
| [`EMA.get_state_dict`](../saddlellm/AdvancedTechniques.py#L108)<br><sub>`get_state_dict(self) -> Dict`</sub> | method | 获取 EMA 状态字典 (用于 checkpoint)。 | `v.clone`, `self.shadow.items` |
| [`EMA.load_state_dict`](../saddlellm/AdvancedTechniques.py#L116)<br><sub>`load_state_dict(self, state: Dict)`</sub> | method | 加载 EMA 状态。 | `items`, `copy_`, `state.get` |
| [`EMA._get_model`](../saddlellm/AdvancedTechniques.py#L123)<br><sub>`_get_model(self)`</sub> | method | 通过 shadow 参数反查模型。 | `gc.get_objects`, `isinstance`, `obj.named_parameters`, `RuntimeError` |
| [`EMACallback.__init__`](../saddlellm/AdvancedTechniques.py#L150)<br><sub>`__init__(self, model: nn.Module, decay: float=0.999)`</sub> | method | 初始化 `EMACallback` 实例及其运行依赖。 | `EMA` |
| [`EMACallback.on_optimizer_step`](../saddlellm/AdvancedTechniques.py#L154)<br><sub>`on_optimizer_step(self)`</sub> | method | `EMACallback` 中实现`on_optimizer_step`的公开操作。 | `self.ema.update` |
| [`EMACallback.on_eval_begin`](../saddlellm/AdvancedTechniques.py#L157)<br><sub>`on_eval_begin(self)`</sub> | method | `EMACallback` 中实现`on_eval_begin`的公开操作。 | `self.ema.apply` |
| [`EMACallback.on_eval_end`](../saddlellm/AdvancedTechniques.py#L160)<br><sub>`on_eval_end(self)`</sub> | method | `EMACallback` 中实现`on_eval_end`的公开操作。 | `self.ema.restore` |
| [`EMACallback.on_save_checkpoint`](../saddlellm/AdvancedTechniques.py#L163)<br><sub>`on_save_checkpoint(self, path: str, tokenizer=None)`</sub> | method | `EMACallback` 中保存检查点的公开操作。 | `self.ema.apply_and_save` |
| [`EMACallback.state_dict`](../saddlellm/AdvancedTechniques.py#L166)<br><sub>`state_dict(self)`</sub> | method | `EMACallback` 中实现状态的公开操作。 | `self.ema.get_state_dict` |
| [`EMACallback.load_state_dict`](../saddlellm/AdvancedTechniques.py#L169)<br><sub>`load_state_dict(self, state)`</sub> | method | `EMACallback` 中加载状态的公开操作。 | `self.ema.load_state_dict` |
| [`SelfConsistency.__init__`](../saddlellm/AdvancedTechniques.py#L206)<br><sub>`__init__(self, model, tokenizer, num_samples: int=8, temperature: float=0.7)`</sub> | method | 初始化 `SelfConsistency` 实例及其运行依赖。 | `next`, `model.parameters` |
| [`SelfConsistency.solve`](../saddlellm/AdvancedTechniques.py#L214)<br><sub>`solve(self, question: str, num_samples: int=None, extract_answer_fn: Optional[Callable]=None, cot_prompt: str='让我们一步步思考。') -> Dict`</sub> | method | 自洽性求解。 | `self._generate_n`, `extractor`, `self._normalize`, `answer_counts.get`, `max`, `len`, `round`, `dict`, `sorted`, `answer_counts.items` |
| [`SelfConsistency._generate_n`](../saddlellm/AdvancedTechniques.py#L264)<br><sub>`_generate_n(self, prompt: str, n: int) -> List[str]`</sub> | method | 生成 N 条不同采样。 | `self.model.eval`, `to`, `self.tokenizer`, `range`, `self.model.generate`, `self.tokenizer.decode`, `len`, `results.append`, `response.strip` |
| [`SelfConsistency._default_extractor`](../saddlellm/AdvancedTechniques.py#L288)<br><sub>`_default_extractor(self, text: str) -> str`</sub> | method | 默认答案提取: 找"答案"之后的数字或关键词。 | `re.search`, `strip`, `m.group`, `re.split`, `reversed`, `s.strip`, `len` |
| [`SelfConsistency._normalize`](../saddlellm/AdvancedTechniques.py#L312)<br><sub>`_normalize(self, answer: str) -> str`</sub> | method | 标准化答案: 去空格、统一符号。 | `lower`, `answer.strip`, `re.sub`, `replace`, `answer.replace`, `answer.startswith`, `len` |
| [`EvolInstruct.__init__`](../saddlellm/AdvancedTechniques.py#L405)<br><sub>`__init__(self, model, tokenizer, temperature: float=0.8)`</sub> | method | 初始化 `EvolInstruct` 实例及其运行依赖。 | `next`, `model.parameters` |
| [`EvolInstruct.evolve`](../saddlellm/AdvancedTechniques.py#L411)<br><sub>`evolve(self, instruction: str, evolve_types: List[str]=None) -> List[str]`</sub> | method | 将一条指令进化为多条复杂版本。 | `list`, `self.EVOLVE_PROMPTS.keys`, `format`, `self._generate`, `results.append` |
| [`EvolInstruct.evolve_batch`](../saddlellm/AdvancedTechniques.py#L427)<br><sub>`evolve_batch(self, seed_instructions: List[str], rounds: int=3, keep_original: bool=True) -> List[str]`</sub> | method | 批量进化: 多轮深度进化。 | `list`, `range`, `logger.info`, `len`, `random.sample`, `self.EVOLVE_PROMPTS.keys`, `self.evolve`, `new_instructions.extend`, `pool.extend`, `dict.fromkeys` |
| [`EvolInstruct._generate`](../saddlellm/AdvancedTechniques.py#L461)<br><sub>`_generate(self, prompt: str) -> str`</sub> | method | `EvolInstruct` 中生成`generate`的内部辅助逻辑。 | `self.model.eval`, `to`, `self.tokenizer`, `len`, `self.tokenizer.decode`, `torch.no_grad`, `self.model.generate`, `strip` |
| [`ModelSoup.average_checkpoints`](../saddlellm/AdvancedTechniques.py#L511)<br><sub>`average_checkpoints(self, checkpoint_paths: List[str], output_path: str, method: str='uniform', val_fn: Optional[Callable]=None) -> None`</sub> | method | 平均多个检查点并保存。 | `self._uniform_average`, `ValueError`, `self._greedy_average`, `self._learned_average`, `logger.info` |
| [`ModelSoup._uniform_average`](../saddlellm/AdvancedTechniques.py#L542)<br><sub>`_uniform_average(self, paths: List[str], output: str)`</sub> | method | 均匀平均所有检查点。 | `AutoModelForCausalLM.from_pretrained`, `state_dicts.append`, `model.state_dict`, `float`, `to`, `mean`, `torch.stack`, `base.load_state_dict`, `base.save_pretrained` |
| [`ModelSoup._greedy_average`](../saddlellm/AdvancedTechniques.py#L563)<br><sub>`_greedy_average(self, paths: List[str], output: str, val_fn: Callable)`</sub> | method | 贪心选择检查点加入"汤"。 依次尝试加入每个检查点, 只有验证分数提升才保留。 | `self._uniform_average`, `val_fn`, `os.path.exists`, `selected.append`, `logger.info` |
| [`ModelSoup._learned_average`](../saddlellm/AdvancedTechniques.py#L591)<br><sub>`_learned_average(self, paths: List[str], output: str, val_fn: Callable)`</sub> | method | 通过简单网格搜索学习最佳权重 (简化版)。 对于少量检查点 (≤5), 尝试不同的权重组合。 | `len`, `self._uniform_average`, `AutoModelForCausalLM.from_pretrained`, `m.state_dict`, `float`, `weight_combinations.append`, `sum`, `zip`, `weighted.to`, `base_model.load_state_dict` |
| [`InstructionBacktranslation.__init__`](../saddlellm/AdvancedTechniques.py#L715)<br><sub>`__init__(self, model, tokenizer, temperature: float=0.7)`</sub> | method | 初始化 `InstructionBacktranslation` 实例及其运行依赖。 | `next`, `model.parameters` |
| [`InstructionBacktranslation.backtranslate`](../saddlellm/AdvancedTechniques.py#L721)<br><sub>`backtranslate(self, documents: List[str], quality_filter: bool=True, min_quality: int=3, max_pairs: int=1000) -> List[Dict]`</sub> | method | 从文档自动生成 instruction-output 对。 | `enumerate`, `len`, `self.BACKTRANSLATE_PROMPT.format`, `strip`, `self._generate`, `re.sub`, `self._check_quality`, `pairs.append`, `logger.info`, `pair.get` |
| [`InstructionBacktranslation._check_quality`](../saddlellm/AdvancedTechniques.py#L770)<br><sub>`_check_quality(self, pair: Dict) -> int`</sub> | method | 让模型自检质量。 | `self.QUALITY_CHECK_PROMPT.format`, `strip`, `self._generate`, `re.search`, `int`, `match.group` |
| [`InstructionBacktranslation._generate`](../saddlellm/AdvancedTechniques.py#L784)<br><sub>`_generate(self, prompt: str) -> str`</sub> | method | `InstructionBacktranslation` 中生成`generate`的内部辅助逻辑。 | `self.model.eval`, `to`, `self.tokenizer`, `len`, `self.tokenizer.decode`, `torch.no_grad`, `self.model.generate`, `strip` |
| [`create_improvement_pipeline`](../saddlellm/AdvancedTechniques.py#L808)<br><sub>`create_improvement_pipeline(model, tokenizer, seed_data: List[str]) -> Dict`</sub> | function | 组合使用 5 种技术提升模型效果: | `logger.info`, `EvolInstruct`, `evolver.evolve_batch`, `len`, `SelfConsistency`, `sc.solve`, `verified.append`, `min` |

## `saddlellm/AgentModelProvider.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`_separate_inline_reasoning`](../saddlellm/AgentModelProvider.py#L37)<br><sub>`_separate_inline_reasoning(message: Mapping[str, Any]) -> Dict[str, Any]`</sub> | function | Keep reasoning available for model continuity but out of user content. | `dict`, `str`, `cleaned.get`, `match.strip`, `_THINK_BLOCK.findall`, `_THINK_BLOCK.sub`, `re.search`, `strip`, `open_tag.end`, `extracted.append` |
| [`_private_provider_address`](../saddlellm/AgentModelProvider.py#L58)<br><sub>`_private_provider_address(address: Any) -> bool`</sub> | function | 模块级实现`private_provider_address`的内部辅助逻辑。 | `any` |
| [`_validate_provider_host`](../saddlellm/AgentModelProvider.py#L62)<br><sub>`_validate_provider_host(hostname: str, *, resolve: bool=False, allow_private: bool=False) -> None`</sub> | function | 模块级校验`validate_provider_host`的内部辅助逻辑。 | `lower`, `strip`, `str`, `ModelProviderError`, `ipaddress.ip_address`, `_private_provider_address`, `socket.getaddrinfo` |
| [`endpoint`](../saddlellm/AgentModelProvider.py#L103)<br><sub>`endpoint(base_url: str, allow_private_network: bool=False) -> str`</sub> | function | Return a validated chat-completions endpoint. | `strip`, `str`, `ModelProviderError`, `urllib.parse.urlparse`, `_validate_provider_host`, `bool`, `parsed.path.rstrip`, `path.endswith`, `urllib.parse.urlunparse` |
| [`_NoRedirectHandler.redirect_request`](../saddlellm/AgentModelProvider.py#L134)<br><sub>`redirect_request(self, req, fp, code, msg, headers, newurl)`</sub> | method | `_NoRedirectHandler` 中实现请求的公开操作。 | — |
| [`_open_provider_request`](../saddlellm/AgentModelProvider.py#L138)<br><sub>`_open_provider_request(request: urllib.request.Request, timeout: int)`</sub> | function | 模块级实现请求的内部辅助逻辑。 | `urllib.request.build_opener`, `_NoRedirectHandler`, `opener.open` |
| [`chat_completion`](../saddlellm/AgentModelProvider.py#L143)<br><sub>`chat_completion(provider: Mapping[str, Any], agent: Mapping[str, Any], messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]=()) -> Dict[str, Any]`</sub> | function | Call one OpenAI-compatible model and return its assistant message. | `bool`, `provider.get`, `endpoint`, `str`, `strip`, `agent.get`, `ModelProviderError`, `urllib.parse.urlparse`, `_validate_provider_host`, `os.environ.get` |

## `saddlellm/Architecture.py`

共 12 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ArchitectureSupport.supports`](../saddlellm/Architecture.py#L43)<br><sub>`supports(self, capability: str) -> bool`</sub> | method | `ArchitectureSupport` 中实现`supports`的公开操作。 | `aliases.get`, `bool`, `getattr` |
| [`ArchitectureSupport.to_dict`](../saddlellm/Architecture.py#L54)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ArchitectureSupport` 转为可序列化字典。 | — |
| [`ArchitectureRegistry.get`](../saddlellm/Architecture.py#L295)<br><sub>`get(name: str) -> ArchitectureSupport`</sub> | method | `ArchitectureRegistry` 中读取`get`的公开操作。 | `ArchitectureRegistry.normalize_name`, `KeyError`, `join`, `sorted` |
| [`ArchitectureRegistry.list_all`](../saddlellm/Architecture.py#L302)<br><sub>`list_all(level: Optional[str]=None) -> List[ArchitectureSupport]`</sub> | method | `ArchitectureRegistry` 中列出`list_all`的公开操作。 | `list`, `ARCHITECTURES.values`, `sorted` |
| [`ArchitectureRegistry.normalize_name`](../saddlellm/Architecture.py#L309)<br><sub>`normalize_name(name: str) -> str`</sub> | method | `ArchitectureRegistry` 中规范化`normalize_name`的公开操作。 | `replace`, `lower`, `ARCHITECTURES.items`, `arch.lower`, `alias.lower`, `alias_map.items` |
| [`ArchitectureRegistry.detect_from_model_name`](../saddlellm/Architecture.py#L324)<br><sub>`detect_from_model_name(model_name_or_path: str) -> ArchitectureSupport`</sub> | method | `ArchitectureRegistry` 中检测模型的公开操作。 | `lower`, `ArchitectureSupport` |
| [`ArchitectureRegistry.detect_from_hf_config`](../saddlellm/Architecture.py#L367)<br><sub>`detect_from_hf_config(model_name_or_path: str, trust_remote_code: bool=True) -> ArchitectureSupport`</sub> | method | `ArchitectureRegistry` 中检测配置的公开操作。 | `AutoConfig.from_pretrained`, `getattr`, `ArchitectureRegistry.get`, `ArchitectureRegistry.detect_from_model_name` |
| [`ArchitectureRegistry.from_model_spec`](../saddlellm/Architecture.py#L377)<br><sub>`from_model_spec(spec) -> ArchitectureSupport`</sub> | method | `ArchitectureRegistry` 中实现模型的公开操作。 | `ArchitectureRegistry.get`, `getattr` |
| [`ArchitectureRegistry.supports`](../saddlellm/Architecture.py#L381)<br><sub>`supports(architecture: str, capability: str) -> bool`</sub> | method | `ArchitectureRegistry` 中实现`supports`的公开操作。 | `supports`, `ArchitectureRegistry.get` |
| [`ArchitectureRegistry.require`](../saddlellm/Architecture.py#L385)<br><sub>`require(architecture: str, capability: str) -> ArchitectureSupport`</sub> | method | `ArchitectureRegistry` 中实现`require`的公开操作。 | `ArchitectureRegistry.get`, `support.supports`, `ValueError` |
| [`ArchitectureRegistry.recommend`](../saddlellm/Architecture.py#L395)<br><sub>`recommend(stage: str) -> List[ArchitectureSupport]`</sub> | method | `ArchitectureRegistry` 中实现`recommend`的公开操作。 | `stage.lower`, `ArchitectureRegistry.list_all`, `arch.supports` |
| [`ArchitectureRegistry.support_matrix`](../saddlellm/Architecture.py#L400)<br><sub>`support_matrix(markdown: bool=True) -> str`</sub> | method | `ArchitectureRegistry` 中实现`support_matrix`的公开操作。 | `ArchitectureRegistry.list_all`, `rows.append`, `join`, `len`, `lines.append`, `replace`, `str` |

## `saddlellm/BackendAdapters.py`

共 9 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`LaunchPlan.to_dict`](../saddlellm/BackendAdapters.py#L21)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `LaunchPlan` 转为可序列化字典。 | `asdict` |
| [`BackendAdapterRegistry.create_launch_plan`](../saddlellm/BackendAdapters.py#L29)<br><sub>`create_launch_plan(config_path: str, backend: str='auto', num_gpus: int=1, num_nodes: int=1, backend_plan_path: Optional[str]=None) -> LaunchPlan`</sub> | method | `BackendAdapterRegistry` 中创建计划的公开操作。 | `BackendAdapterRegistry._infer_backend`, `BackendAdapterRegistry._normalize_backend_name`, `int`, `ValueError`, `BackendAdapterRegistry._accelerate_plan`, `NotImplementedError` |
| [`BackendAdapterRegistry._normalize_backend_name`](../saddlellm/BackendAdapters.py#L54)<br><sub>`_normalize_backend_name(backend: str) -> str`</sub> | method | `BackendAdapterRegistry` 中规范化后端的内部辅助逻辑。 | `aliases.get` |
| [`BackendAdapterRegistry.save_launch_script`](../saddlellm/BackendAdapters.py#L63)<br><sub>`save_launch_script(plan: LaunchPlan, path: str) -> str`</sub> | method | `BackendAdapterRegistry` 中保存`save_launch_script`的公开操作。 | `os.makedirs`, `os.path.dirname`, `plan.env.items`, `lines.append`, `join`, `BackendAdapterRegistry._quote`, `open`, `f.write` |
| [`BackendAdapterRegistry.save_launch_plan`](../saddlellm/BackendAdapters.py#L76)<br><sub>`save_launch_plan(plan: LaunchPlan, path: str) -> str`</sub> | method | `BackendAdapterRegistry` 中保存计划的公开操作。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump`, `plan.to_dict` |
| [`BackendAdapterRegistry._accelerate_plan`](../saddlellm/BackendAdapters.py#L83)<br><sub>`_accelerate_plan(config_path: str, backend: str, num_gpus: int, num_nodes: int=1) -> LaunchPlan`</sub> | method | `BackendAdapterRegistry` 中规划计划的内部辅助逻辑。 | `LaunchPlan`, `ValueError`, `str`, `backend.startswith`, `os.environ.get`, `notes.append` |
| [`BackendAdapterRegistry._hybrid_plan`](../saddlellm/BackendAdapters.py#L128)<br><sub>`_hybrid_plan(config_path: str, backend: str, num_gpus: int, backend_plan_path: Optional[str]) -> LaunchPlan`</sub> | method | `BackendAdapterRegistry` 中规划计划的内部辅助逻辑。 | `str`, `notes.append`, `LaunchPlan` |
| [`BackendAdapterRegistry._infer_backend`](../saddlellm/BackendAdapters.py#L152)<br><sub>`_infer_backend(config_path: str, backend_plan_path: Optional[str]) -> str`</sub> | method | `BackendAdapterRegistry` 中推断后端的内部辅助逻辑。 | `os.path.exists`, `open`, `get`, `json.load`, `endswith`, `config_path.lower`, `yaml.safe_load`, `data.get` |
| [`BackendAdapterRegistry._quote`](../saddlellm/BackendAdapters.py#L168)<br><sub>`_quote(value: str) -> str`</sub> | method | `BackendAdapterRegistry` 中实现`quote`的内部辅助逻辑。 | `any`, `ch.isspace`, `value.replace` |

## `saddlellm/BenchmarkRunner.py`

共 17 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`BenchmarkRunner.__init__`](../saddlellm/BenchmarkRunner.py#L56)<br><sub>`__init__(self, model_path: str, tasks: Optional[List[str]]=None, max_samples: int=1000, batch_size: int=8, device: str='auto', tokenizer_path: Optional[str]=None)`</sub> | method | 初始化 `BenchmarkRunner` 实例及其运行依赖。 | — |
| [`BenchmarkRunner.run`](../saddlellm/BenchmarkRunner.py#L75)<br><sub>`run(self) -> List[EvalResult]`</sub> | method | 运行所有评测任务 | `self._load_model`, `logger.info`, `time.time`, `self._eval_perplexity`, `self._eval_hellaswag`, `self._eval_mmlu`, `self._eval_arc`, `self._eval_winogrande`, `self._eval_piqa`, `self._eval_boolq` |
| [`BenchmarkRunner._eval_perplexity`](../saddlellm/BenchmarkRunner.py#L136)<br><sub>`_eval_perplexity(self) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_perplexity`的内部辅助逻辑。 | `load_dataset`, `self._model.eval`, `torch.no_grad`, `enumerate`, `example.get`, `len`, `text.strip`, `self._tokenizer`, `v.to`, `inputs.items` |
| [`BenchmarkRunner._eval_hellaswag`](../saddlellm/BenchmarkRunner.py#L167)<br><sub>`_eval_hellaswag(self) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_hellaswag`的内部辅助逻辑。 | `self._eval_hf_multiple_choice` |
| [`BenchmarkRunner._eval_mmlu`](../saddlellm/BenchmarkRunner.py#L170)<br><sub>`_eval_mmlu(self) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_mmlu`的内部辅助逻辑。 | `load_dataset`, `self._eval_hf_multiple_choice_ds`, `scores.append`, `sum`, `len` |
| [`BenchmarkRunner._eval_arc`](../saddlellm/BenchmarkRunner.py#L188)<br><sub>`_eval_arc(self, config: str) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_arc`的内部辅助逻辑。 | `self._eval_hf_multiple_choice` |
| [`BenchmarkRunner._eval_winogrande`](../saddlellm/BenchmarkRunner.py#L191)<br><sub>`_eval_winogrande(self) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_winogrande`的内部辅助逻辑。 | `self._eval_hf_multiple_choice` |
| [`BenchmarkRunner._eval_piqa`](../saddlellm/BenchmarkRunner.py#L194)<br><sub>`_eval_piqa(self) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_piqa`的内部辅助逻辑。 | `self._eval_hf_multiple_choice` |
| [`BenchmarkRunner._eval_boolq`](../saddlellm/BenchmarkRunner.py#L197)<br><sub>`_eval_boolq(self) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_boolq`的内部辅助逻辑。 | `self._eval_hf_multiple_choice` |
| [`BenchmarkRunner._eval_lambada`](../saddlellm/BenchmarkRunner.py#L200)<br><sub>`_eval_lambada(self) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_lambada`的内部辅助逻辑。 | `self._eval_hf_multiple_choice` |
| [`BenchmarkRunner._eval_gsm8k`](../saddlellm/BenchmarkRunner.py#L203)<br><sub>`_eval_gsm8k(self) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_gsm8k`的内部辅助逻辑。 | `self._eval_perplexity` |
| [`BenchmarkRunner._eval_hf_multiple_choice`](../saddlellm/BenchmarkRunner.py#L206)<br><sub>`_eval_hf_multiple_choice(self, dataset_name: str, config: str) -> float`</sub> | method | 通用多选题评测 | `load_dataset`, `self._eval_hf_multiple_choice_ds` |
| [`BenchmarkRunner._eval_hf_multiple_choice_ds`](../saddlellm/BenchmarkRunner.py#L222)<br><sub>`_eval_hf_multiple_choice_ds(self, dataset) -> float`</sub> | method | `BenchmarkRunner` 中实现`eval_hf_multiple_choice_ds`的内部辅助逻辑。 | `self._model.eval`, `torch.no_grad`, `example.get`, `isinstance`, `e.get`, `str`, `self._tokenizer`, `v.to`, `inputs.items`, `self._model` |
| [`BenchmarkRunner._load_model`](../saddlellm/BenchmarkRunner.py#L266)<br><sub>`_load_model(self)`</sub> | method | `BenchmarkRunner` 中加载模型的内部辅助逻辑。 | `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, `torch.cuda.is_available`, `self._model.cpu` |
| [`BenchmarkRunner.save_results`](../saddlellm/BenchmarkRunner.py#L285)<br><sub>`save_results(self, results: List[EvalResult], path: str)`</sub> | method | `BenchmarkRunner` 中保存`save_results`的公开操作。 | `open`, `json.dump` |
| [`BenchmarkRunner.load_results`](../saddlellm/BenchmarkRunner.py#L290)<br><sub>`load_results(cls, path: str) -> List[EvalResult]`</sub> | method | `BenchmarkRunner` 中加载`load_results`的公开操作。 | `open`, `json.load`, `EvalResult` |
| [`BenchmarkRunner.compare`](../saddlellm/BenchmarkRunner.py#L296)<br><sub>`compare(result_paths: List[str], output_path: Optional[str]=None) -> str`</sub> | method | 比较多个模型的结果,生成对比表格 | `os.path.exists`, `open`, `all_results.append`, `json.load`, `set`, `all_tasks.add`, `sorted`, `enumerate`, `get`, `lines.append` |

## `saddlellm/BuiltinMediaCodecs.py`

共 31 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TextConditionEncoder.dimension`](../saddlellm/BuiltinMediaCodecs.py#L29)<br><sub>`dimension(self) -> int`</sub> | method | `TextConditionEncoder` 中实现`dimension`的公开操作。 | — |
| [`TextConditionEncoder.encode`](../saddlellm/BuiltinMediaCodecs.py#L33)<br><sub>`encode(self, texts: Sequence[str]) -> np.ndarray`</sub> | method | `TextConditionEncoder` 中编码`encode`的公开操作。 | — |
| [`TextConditionEncoder.fingerprint`](../saddlellm/BuiltinMediaCodecs.py#L37)<br><sub>`fingerprint(self) -> Dict[str, Any]`</sub> | method | `TextConditionEncoder` 中实现`fingerprint`的公开操作。 | — |
| [`HashTextConditionEncoder.__init__`](../saddlellm/BuiltinMediaCodecs.py#L44)<br><sub>`__init__(self, dimension: int=256, ngram: int=3, seed: int=0) -> None`</sub> | method | 初始化 `HashTextConditionEncoder` 实例及其运行依赖。 | `ValueError`, `int` |
| [`HashTextConditionEncoder.dimension`](../saddlellm/BuiltinMediaCodecs.py#L52)<br><sub>`dimension(self) -> int`</sub> | method | `HashTextConditionEncoder` 中实现`dimension`的公开操作。 | — |
| [`HashTextConditionEncoder.encode`](../saddlellm/BuiltinMediaCodecs.py#L55)<br><sub>`encode(self, texts: Sequence[str]) -> np.ndarray`</sub> | method | `HashTextConditionEncoder` 中编码`encode`的公开操作。 | `np.zeros`, `len`, `self.seed.to_bytes`, `enumerate`, `encode`, `str`, `range`, `max`, `digest`, `hashlib.blake2b` |
| [`HashTextConditionEncoder.fingerprint`](../saddlellm/BuiltinMediaCodecs.py#L73)<br><sub>`fingerprint(self) -> Dict[str, Any]`</sub> | method | `HashTextConditionEncoder` 中实现`fingerprint`的公开操作。 | — |
| [`HuggingFaceTextConditionEncoder.__init__`](../saddlellm/BuiltinMediaCodecs.py#L86)<br><sub>`__init__(self, model_name_or_path: str, *, device: str='auto', max_length: int=256, local_files_only: bool=False, trust_remote_code: bool=False) -> None`</sub> | method | 初始化 `HuggingFaceTextConditionEncoder` 实例及其运行依赖。 | `ValueError`, `int`, `bool`, `torch.device`, `torch.cuda.is_available`, `AutoTokenizer.from_pretrained`, `eval`, `to`, `AutoModel.from_pretrained`, `getattr` |
| [`HuggingFaceTextConditionEncoder.dimension`](../saddlellm/BuiltinMediaCodecs.py#L126)<br><sub>`dimension(self) -> int`</sub> | method | `HuggingFaceTextConditionEncoder` 中实现`dimension`的公开操作。 | — |
| [`HuggingFaceTextConditionEncoder.encode`](../saddlellm/BuiltinMediaCodecs.py#L129)<br><sub>`encode(self, texts: Sequence[str]) -> np.ndarray`</sub> | method | `HuggingFaceTextConditionEncoder` 中编码`encode`的公开操作。 | `self.tokenizer`, `list`, `value.to`, `batch.items`, `torch.inference_mode`, `self.model`, `batch.get`, `torch.ones`, `sum`, `clamp_min` |
| [`HuggingFaceTextConditionEncoder.fingerprint`](../saddlellm/BuiltinMediaCodecs.py#L147)<br><sub>`fingerprint(self) -> Dict[str, Any]`</sub> | method | `HuggingFaceTextConditionEncoder` 中实现`fingerprint`的公开操作。 | — |
| [`build_text_condition_encoder`](../saddlellm/BuiltinMediaCodecs.py#L157)<br><sub>`build_text_condition_encoder(name: str='hash', **config: Any) -> TextConditionEncoder`</sub> | function | 模块级构建`build_text_condition_encoder`的公开操作。 | `lower`, `strip`, `str`, `HashTextConditionEncoder`, `HuggingFaceTextConditionEncoder`, `ValueError` |
| [`RGBImageCodec.__init__`](../saddlellm/BuiltinMediaCodecs.py#L180)<br><sub>`__init__(self, height: int=32, width: int=32) -> None`</sub> | method | 初始化 `RGBImageCodec` 实例及其运行依赖。 | `ValueError`, `int` |
| [`RGBImageCodec.encode`](../saddlellm/BuiltinMediaCodecs.py#L186)<br><sub>`encode(self, value: Any, **kwargs: Any) -> np.ndarray`</sub> | method | `RGBImageCodec` 中编码`encode`的公开操作。 | `isinstance`, `Image.open`, `os.fspath`, `resize`, `image.convert`, `np.asarray`, `np.transpose` |
| [`RGBImageCodec.decode`](../saddlellm/BuiltinMediaCodecs.py#L194)<br><sub>`decode(self, latents: Any, **kwargs: Any)`</sub> | method | `RGBImageCodec` 中解码`decode`的公开操作。 | `np.asarray`, `ValueError`, `np.clip`, `np.transpose`, `Image.fromarray`, `pixels.astype`, `kwargs.get`, `image.save` |
| [`RGBImageCodec.fingerprint`](../saddlellm/BuiltinMediaCodecs.py#L209)<br><sub>`fingerprint(self) -> Dict[str, Any]`</sub> | method | `RGBImageCodec` 中实现`fingerprint`的公开操作。 | `fingerprint`, `super` |
| [`WAVResidualCodec.__init__`](../saddlellm/BuiltinMediaCodecs.py#L225)<br><sub>`__init__(self, sample_rate: int=16000, frame_size: int=320, max_frames: int=256, num_codebooks: int=4, codebook_size: int=256) -> None`</sub> | method | 初始化 `WAVResidualCodec` 实例及其运行依赖。 | `items`, `int`, `ValueError` |
| [`WAVResidualCodec.encode`](../saddlellm/BuiltinMediaCodecs.py#L250)<br><sub>`encode(self, value: Any, **kwargs: Any) -> np.ndarray`</sub> | method | `WAVResidualCodec` 中编码`encode`的公开操作。 | `self.encode_with_attention_mask` |
| [`WAVResidualCodec.encode_with_attention_mask`](../saddlellm/BuiltinMediaCodecs.py#L254)<br><sub>`encode_with_attention_mask(self, value: Any) -> tuple[np.ndarray, np.ndarray]`</sub> | method | `WAVResidualCodec` 中编码注意力的公开操作。 | `_read_wav_mono`, `os.fspath`, `_resample_linear`, `ValueError`, `np.zeros`, `min`, `int`, `math.ceil`, `mean`, `padded.reshape` |
| [`WAVResidualCodec.decode`](../saddlellm/BuiltinMediaCodecs.py#L280)<br><sub>`decode(self, latents: Any, **kwargs: Any) -> np.ndarray`</sub> | method | `WAVResidualCodec` 中解码`decode`的公开操作。 | `np.asarray`, `ValueError`, `codes.min`, `codes.max`, `np.zeros`, `codebook.astype`, `np.repeat`, `np.clip`, `kwargs.get`, `_write_wav_mono` |
| [`WAVResidualCodec.fingerprint`](../saddlellm/BuiltinMediaCodecs.py#L300)<br><sub>`fingerprint(self) -> Dict[str, Any]`</sub> | method | `WAVResidualCodec` 中实现`fingerprint`的公开操作。 | `fingerprint`, `super` |
| [`FrameVideoCodec.__init__`](../saddlellm/BuiltinMediaCodecs.py#L323)<br><sub>`__init__(self, frames: int=8, height: int=32, width: int=32) -> None`</sub> | method | 初始化 `FrameVideoCodec` 实例及其运行依赖。 | `ValueError`, `int` |
| [`FrameVideoCodec.encode`](../saddlellm/BuiltinMediaCodecs.py#L330)<br><sub>`encode(self, value: Any, **kwargs: Any) -> np.ndarray`</sub> | method | `FrameVideoCodec` 中编码`encode`的公开操作。 | `_load_video_frames`, `os.fspath`, `ValueError`, `astype`, `round`, `np.linspace`, `len`, `resize`, `convert`, `int` |
| [`FrameVideoCodec.decode`](../saddlellm/BuiltinMediaCodecs.py#L344)<br><sub>`decode(self, latents: Any, **kwargs: Any) -> List[Any]`</sub> | method | `FrameVideoCodec` 中解码`decode`的公开操作。 | `np.asarray`, `ValueError`, `range`, `np.clip`, `np.transpose`, `frames.append`, `Image.fromarray`, `pixels.astype`, `kwargs.get`, `save` |
| [`FrameVideoCodec.fingerprint`](../saddlellm/BuiltinMediaCodecs.py#L366)<br><sub>`fingerprint(self) -> Dict[str, Any]`</sub> | method | `FrameVideoCodec` 中实现`fingerprint`的公开操作。 | `fingerprint`, `super` |
| [`image_resampling`](../saddlellm/BuiltinMediaCodecs.py#L375)<br><sub>`image_resampling()`</sub> | function | 模块级实现图像的公开操作。 | — |
| [`register_builtin_media_codecs`](../saddlellm/BuiltinMediaCodecs.py#L381)<br><sub>`register_builtin_media_codecs() -> None`</sub> | function | Idempotently register the dependency-light baseline implementations. | `ModalityCodecRegistry.list_specs`, `ModalityCodecRegistry.register` |
| [`_read_wav_mono`](../saddlellm/BuiltinMediaCodecs.py#L395)<br><sub>`_read_wav_mono(path: str) -> tuple[np.ndarray, int]`</sub> | function | 模块级读取`read_wav_mono`的内部辅助逻辑。 | `wave.open`, `stream.getnchannels`, `stream.getsampwidth`, `stream.getframerate`, `stream.readframes`, `stream.getnframes`, `astype`, `np.frombuffer`, `reshape`, `np.where` |
| [`_resample_linear`](../saddlellm/BuiltinMediaCodecs.py#L419)<br><sub>`_resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray`</sub> | function | 模块级实现`resample_linear`的内部辅助逻辑。 | `max`, `int`, `round`, `np.linspace`, `astype`, `np.interp` |
| [`_write_wav_mono`](../saddlellm/BuiltinMediaCodecs.py#L428)<br><sub>`_write_wav_mono(path: str, samples: np.ndarray, sample_rate: int) -> None`</sub> | function | 模块级写入`write_wav_mono`的内部辅助逻辑。 | `parent.mkdir`, `Path`, `np.clip`, `astype`, `wave.open`, `stream.setnchannels`, `stream.setsampwidth`, `stream.setframerate`, `stream.writeframes`, `pcm.tobytes` |
| [`_load_video_frames`](../saddlellm/BuiltinMediaCodecs.py#L439)<br><sub>`_load_video_frames(path: str) -> List[Any]`</sub> | function | 模块级加载`load_video_frames`的内部辅助逻辑。 | `Path`, `source.is_dir`, `copy`, `Image.open`, `sorted`, `source.iterdir`, `item.is_file`, `item.suffix.lower`, `frame.copy`, `ImageSequence.Iterator` |

## `saddlellm/ChatUI.py`

共 4 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ChatUI.launch`](../saddlellm/ChatUI.py#L30)<br><sub>`launch(model, tokenizer=None, title: str='SaddleLLM Chat', port: int=7860, share: bool=False, use_cot: bool=True, max_tokens: int=1024)`</sub> | method | 启动 Gradio Web 聊天界面。 | `AutoTokenizer.from_pretrained`, `next`, `model.parameters`, `gr.Blocks`, `gr.themes.Soft`, `gr.Markdown`, `gr.Chatbot`, `gr.Textbox`, `gr.Button`, `gr.Row` |
| [`ChatUI.launch.chat_fn`](../saddlellm/ChatUI.py#L56)<br><sub>`chat_fn(message, history_list, temperature, top_p, do_cot)`</sub> | nested function | `ChatUI` 中实现`chat_fn`的局部回调/辅助逻辑。 | `model.eval`, `to`, `tokenizer`, `torch.no_grad`, `model.generate`, `tokenizer.decode`, `len`, `strip`, `history_list.append` |
| [`ChatUI.terminal`](../saddlellm/ChatUI.py#L122)<br><sub>`terminal(model, tokenizer=None, use_cot: bool=True)`</sub> | method | 终端聊天模式 (无额外依赖)。 | `AutoTokenizer.from_pretrained`, `next`, `model.parameters`, `print`, `strip`, `input`, `model.eval`, `to`, `tokenizer`, `torch.no_grad` |
| [`_patch_easy`](../saddlellm/ChatUI.py#L202)<br><sub>`_patch_easy()`</sub> | function | 给 easy 模块添加 chat_ui 函数。 | `ChatUI.launch`, `ChatUI.terminal` |

## `saddlellm/CurriculumScheduler.py`

共 9 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`CurriculumScheduler.__init__`](../saddlellm/CurriculumScheduler.py#L40)<br><sub>`__init__(self, stages: List[StageConfig])`</sub> | method | 初始化 `CurriculumScheduler` 实例及其运行依赖。 | `sorted`, `self._validate_stages` |
| [`CurriculumScheduler.standard_curriculum`](../saddlellm/CurriculumScheduler.py#L46)<br><sub>`standard_curriculum(cls, total_steps: int, max_seq_length: int=2048, base_lr: float=0.0003) -> 'CurriculumScheduler'`</sub> | method | 创建标准 4 阶段课程表 | `int`, `StageConfig`, `min`, `cls` |
| [`CurriculumScheduler.get_current_config`](../saddlellm/CurriculumScheduler.py#L103)<br><sub>`get_current_config(self, step: int) -> StageConfig`</sub> | method | 获取当前步数对应的阶段配置 | `enumerate` |
| [`CurriculumScheduler.is_stage_transition`](../saddlellm/CurriculumScheduler.py#L112)<br><sub>`is_stage_transition(self, step: int) -> bool`</sub> | method | 检查当前步数是否是阶段切换点 | — |
| [`CurriculumScheduler.get_stage`](../saddlellm/CurriculumScheduler.py#L119)<br><sub>`get_stage(self, step: int) -> Optional[StageConfig]`</sub> | method | `CurriculumScheduler` 中读取训练阶段的公开操作。 | `self.get_current_config` |
| [`CurriculumScheduler.current_stage_name`](../saddlellm/CurriculumScheduler.py#L123)<br><sub>`current_stage_name(self) -> str`</sub> | method | `CurriculumScheduler` 中实现训练阶段的公开操作。 | — |
| [`CurriculumScheduler.get_progress`](../saddlellm/CurriculumScheduler.py#L126)<br><sub>`get_progress(self, step: int) -> Dict[str, Any]`</sub> | method | 获取训练进度摘要 | `self.get_current_config`, `max` |
| [`CurriculumScheduler.plot_curriculum`](../saddlellm/CurriculumScheduler.py#L143)<br><sub>`plot_curriculum(self, save_path: Optional[str]=None)`</sub> | method | 可视化课程表 | `logger.warning`, `np.arange`, `np.array`, `self.get_current_config`, `plt.subplots`, `ax1.plot`, `ax1.set_ylabel`, `ax1.set_title`, `ax1.grid`, `ax2.plot` |
| [`CurriculumScheduler._validate_stages`](../saddlellm/CurriculumScheduler.py#L187)<br><sub>`_validate_stages(self)`</sub> | method | 验证阶段配置没有空隙或重叠 | `range`, `len`, `logger.warning` |

## `saddlellm/DataBalance.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`BalancedPretrainer.__init__`](../saddlellm/DataBalance.py#L34)<br><sub>`__init__(self, args)`</sub> | method | 初始化 `BalancedPretrainer` 实例及其运行依赖。 | `AutoTokenizer.from_pretrained`, `self._load_datasets`, `self._init_domain_weights`, `defaultdict` |
| [`BalancedPretrainer._load_datasets`](../saddlellm/DataBalance.py#L41)<br><sub>`_load_datasets(self) -> List[DatasetConfig]`</sub> | method | 加载所有数据集配置 | `exists`, `Path`, `ValueError`, `datasets.append`, `DatasetConfig`, `logger.info`, `len` |
| [`BalancedPretrainer._init_domain_weights`](../saddlellm/DataBalance.py#L57)<br><sub>`_init_domain_weights(self) -> Dict[str, float]`</sub> | method | 初始化领域权重 | `sum` |
| [`BalancedPretrainer._adaptive_sampling`](../saddlellm/DataBalance.py#L65)<br><sub>`_adaptive_sampling(self, epoch: int) -> Dict[str, float]`</sub> | method | 自适应采样策略核心算法 | `len`, `np.diff`, `np.mean`, `sum`, `abs`, `domain_loss_changes.values`, `self.domain_weights.items`, `adjusted_weights.values`, `adjusted_weights.items` |
| [`BalancedPretrainer._load_data_chunk`](../saddlellm/DataBalance.py#L100)<br><sub>`_load_data_chunk(self, dataset: DatasetConfig, chunk_size: int=10000)`</sub> | method | 加载数据块 | `open`, `range`, `f.readline`, `lines.append`, `dataset.path.endswith`, `json.loads`, `line.strip` |
| [`BalancedPretrainer.train`](../saddlellm/DataBalance.py#L112)<br><sub>`train(self)`</sub> | method | 执行训练 | `logger.info`, `range`, `self._adaptive_sampling`, `self._load_data_chunk`, `DataLoader`, `int`, `self._train_epoch`, `self._save_checkpoint` |
| [`BalancedPretrainer._train_epoch`](../saddlellm/DataBalance.py#L142)<br><sub>`_train_epoch(self, epoch: int, dataloaders: Dict, domain_ratios: Dict)`</sub> | method | 训练单个epoch | `range`, `np.random.choice`, `list`, `domain_ratios.keys`, `domain_ratios.values`, `set`, `next`, `iter`, `self._train_step`, `append` |
| [`BalancedPretrainer._train_step`](../saddlellm/DataBalance.py#L171)<br><sub>`_train_step(self, batch, domain: str) -> float`</sub> | method | 模拟训练步骤 | `np.random.uniform` |
| [`BalancedPretrainer._save_checkpoint`](../saddlellm/DataBalance.py#L176)<br><sub>`_save_checkpoint(self, epoch: int)`</sub> | method | 保存检查点 | `Path`, `checkpoint_dir.mkdir`, `logger.info` |
| [`parse_args`](../saddlellm/DataBalance.py#L186)<br><sub>`parse_args()`</sub> | function | 模块级解析`parse_args`的公开操作。 | `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args` |
| [`main`](../saddlellm/DataBalance.py#L223)<br><sub>`main()`</sub> | function | 模块级实现`main`的公开操作。 | `parse_args`, `BalancedPretrainer`, `trainer.train` |

## `saddlellm/DataCatalog.py`

共 12 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DataCatalog.list_all`](../saddlellm/DataCatalog.py#L430)<br><sub>`list_all(category: str=None) -> List[DatasetEntry]`</sub> | method | 列出所有(或按分类筛选)数据集。 | `list`, `CLASSIC_DATASETS.values`, `sorted` |
| [`DataCatalog.get`](../saddlellm/DataCatalog.py#L438)<br><sub>`get(name: str) -> DatasetEntry`</sub> | method | `DataCatalog` 中读取`get`的公开操作。 | `name.lower`, `k.lower`, `KeyError` |
| [`DataCatalog.recommend`](../saddlellm/DataCatalog.py#L446)<br><sub>`recommend(model_size: str='small') -> List[DatasetEntry]`</sub> | method | 根据模型大小推荐数据集。 | `thresholds.get`, `order.index`, `CLASSIC_DATASETS.values`, `recommended.append`, `sorted` |
| [`DataCatalog.fetch`](../saddlellm/DataCatalog.py#L471)<br><sub>`fetch(name: str, streaming: bool=True, split: str=None, max_samples: int=None)`</sub> | method | 拉取数据集 (默认 streaming, 不占磁盘)。 | `DataCatalog.get`, `load_dataset`, `RuntimeError`, `dataset.take`, `dataset.select`, `range`, `min`, `len` |
| [`DataCatalog.small_model_pack`](../saddlellm/DataCatalog.py#L513)<br><sub>`small_model_pack(lang: str='zh', total_tokens_target: int=2000000000, streaming: bool=True)`</sub> | method | 一键获取小模型训练数据包。 | `logger.info`, `DataCatalog.fetch`, `all_ds.append`, `all_weights.append`, `logger.warning`, `RuntimeError`, `sum`, `interleave_datasets`, `dict`, `zip` |
| [`DataCatalog.print_catalog`](../saddlellm/DataCatalog.py#L588)<br><sub>`print_catalog()`</sub> | method | 打印完整数据目录。 | `DataCatalog.list_all`, `print`, `cat.upper`, `len` |
| [`DataQualityClassifier.__init__`](../saddlellm/DataCatalog.py#L630)<br><sub>`__init__(self, lang: str='auto', min_score: float=0.5)`</sub> | method | 初始化 `DataQualityClassifier` 实例及其运行依赖。 | — |
| [`DataQualityClassifier.score`](../saddlellm/DataCatalog.py#L634)<br><sub>`score(self, text: str) -> float`</sub> | method | 对单条文本打分 (0-1)。 | `isinstance`, `text.strip`, `len`, `any`, `re.findall`, `max`, `text.split`, `min`, `set`, `round` |
| [`DataQualityClassifier.is_quality`](../saddlellm/DataCatalog.py#L705)<br><sub>`is_quality(self, text: str) -> bool`</sub> | method | `DataQualityClassifier` 中实现`is_quality`的公开操作。 | `self.score` |
| [`DataQualityClassifier.filter`](../saddlellm/DataCatalog.py#L708)<br><sub>`filter(self, texts: List[str]) -> List[str]`</sub> | method | 过滤低质量文本。 | `self.score` |
| [`DataQualityClassifier.filter_dataset`](../saddlellm/DataCatalog.py#L712)<br><sub>`filter_dataset(self, dataset, text_column: str='text', batch_size: int=1000)`</sub> | method | 过滤 HuggingFace dataset。 | `dataset.filter` |
| [`DataQualityClassifier.filter_dataset._filter`](../saddlellm/DataCatalog.py#L714)<br><sub>`_filter(example)`</sub> | nested function | `DataQualityClassifier` 中过滤`filter`的局部回调/辅助逻辑。 | `example.get`, `self.is_quality` |

## `saddlellm/DataPipeline.py`

共 31 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DataPipeline.__init__`](../saddlellm/DataPipeline.py#L54)<br><sub>`__init__(self, config: PipelineConfig)`</sub> | method | 初始化 `DataPipeline` 实例及其运行依赖。 | `Counter`, `self._validate_config` |
| [`DataPipeline._validate_config`](../saddlellm/DataPipeline.py#L60)<br><sub>`_validate_config(self)`</sub> | method | `DataPipeline` 中校验配置的内部辅助逻辑。 | `ValueError`, `len` |
| [`DataPipeline._ensure_dataset`](../saddlellm/DataPipeline.py#L70)<br><sub>`_ensure_dataset(self)`</sub> | method | `DataPipeline` 中确保数据集的内部辅助逻辑。 | `RuntimeError` |
| [`DataPipeline.collect`](../saddlellm/DataPipeline.py#L74)<br><sub>`collect(self) -> 'DataPipeline'`</sub> | method | 从多个数据源收集文本 | `source.get`, `logger.info`, `load_dataset`, `ds.rename_column`, `all_splits.append`, `any`, `sorted`, `glob.glob`, `os.path.isfile`, `os.path.isdir` |
| [`DataPipeline.clean`](../saddlellm/DataPipeline.py#L164)<br><sub>`clean(self) -> 'DataPipeline'`</sub> | method | 清洗文本: 去除 HTML/URL/特殊字符,利用 TextProcess.py | `self._ensure_dataset`, `TextCleaner`, `logger.warning`, `self._dataset.map`, `self._dataset.filter`, `x.get`, `getattr`, `self._dataset.remove_columns` |
| [`DataPipeline.clean._clean`](../saddlellm/DataPipeline.py#L186)<br><sub>`_clean(example)`</sub> | nested function | `DataPipeline` 中清洗`clean`的局部回调/辅助逻辑。 | `example.get`, `isinstance`, `cleaner.clean_text`, `self._basic_clean_text`, `bool`, `len` |
| [`DataPipeline._basic_clean_text`](../saddlellm/DataPipeline.py#L199)<br><sub>`_basic_clean_text(self, text: str) -> str`</sub> | method | `DataPipeline` 中清洗`basic_clean_text`的内部辅助逻辑。 | `re.sub`, `strip` |
| [`DataPipeline.deduplicate`](../saddlellm/DataPipeline.py#L210)<br><sub>`deduplicate(self, method: Optional[str]=None, threshold: Optional[float]=None) -> 'DataPipeline'`</sub> | method | 文本去重: simhash / minhash / exact | `self._ensure_dataset`, `self._dedup_simhash`, `self._dedup_minhash`, `self._dedup_exact`, `ValueError` |
| [`DataPipeline._dedup_simhash`](../saddlellm/DataPipeline.py#L229)<br><sub>`_dedup_simhash(self, threshold)`</sub> | method | `DataPipeline` 中实现`dedup_simhash`的内部辅助逻辑。 | `logger.warning`, `set`, `self._dataset.filter`, `_is_dup`, `x.get`, `logger.info`, `min`, `len` |
| [`DataPipeline._dedup_simhash._is_dup`](../saddlellm/DataPipeline.py#L238)<br><sub>`_is_dup(text)`</sub> | nested function | `DataPipeline` 中实现`is_dup`的局部回调/辅助逻辑。 | `len`, `Simhash`, `count`, `bin`, `seen.add`, `seen.clear` |
| [`DataPipeline._dedup_minhash`](../saddlellm/DataPipeline.py#L256)<br><sub>`_dedup_minhash(self, threshold)`</sub> | method | `DataPipeline` 中实现`dedup_minhash`的内部辅助逻辑。 | `logger.warning`, `self._dedup_exact`, `MinHashLSH`, `self._dataset.filter`, `_is_dup`, `logger.info` |
| [`DataPipeline._dedup_minhash._tokens`](../saddlellm/DataPipeline.py#L267)<br><sub>`_tokens(text: str)`</sub> | nested function | `DataPipeline` 中实现`tokens`的局部回调/辅助逻辑。 | `text.lower`, `re.search`, `range`, `max`, `len`, `set`, `re.findall` |
| [`DataPipeline._dedup_minhash._signature`](../saddlellm/DataPipeline.py#L273)<br><sub>`_signature(text: str)`</sub> | nested function | `DataPipeline` 中实现`signature`的局部回调/辅助逻辑。 | `MinHash`, `_tokens`, `mh.update`, `token.encode` |
| [`DataPipeline._dedup_minhash._is_dup`](../saddlellm/DataPipeline.py#L279)<br><sub>`_is_dup(example)`</sub> | nested function | `DataPipeline` 中实现`is_dup`的局部回调/辅助逻辑。 | `example.get`, `len`, `_signature`, `lsh.query`, `lsh.insert` |
| [`DataPipeline._dedup_exact`](../saddlellm/DataPipeline.py#L295)<br><sub>`_dedup_exact(self)`</sub> | method | `DataPipeline` 中实现`dedup_exact`的内部辅助逻辑。 | `set`, `self._dataset.filter`, `_is_dup`, `x.get` |
| [`DataPipeline._dedup_exact._is_dup`](../saddlellm/DataPipeline.py#L298)<br><sub>`_is_dup(text)`</sub> | nested function | `DataPipeline` 中实现`is_dup`的局部回调/辅助逻辑。 | `hexdigest`, `hashlib.sha1`, `encode`, `lower`, `text.strip`, `seen.add` |
| [`DataPipeline.filter_quality`](../saddlellm/DataPipeline.py#L309)<br><sub>`filter_quality(self, min_length: Optional[int]=None, lang: Optional[str]=None) -> 'DataPipeline'`</sub> | method | 质量过滤: - 最小文本长度 - 语言检测 - 困惑度/质量评分 - URL/邮箱/手机号过滤 | `self._ensure_dataset`, `DataQualityClassifier`, `logger.warning`, `self._dataset.filter`, `logger.info` |
| [`DataPipeline.filter_quality._filter`](../saddlellm/DataPipeline.py#L329)<br><sub>`_filter(example)`</sub> | nested function | `DataPipeline` 中过滤`filter`的局部回调/辅助逻辑。 | `example.get`, `len`, `text.split`, `max`, `sum`, `c.isupper`, `detect`, `re.search`, `re.findall`, `quality_classifier.is_quality` |
| [`DataPipeline.tokenize_and_pack`](../saddlellm/DataPipeline.py#L380)<br><sub>`tokenize_and_pack(self, tokenizer, text_column: str='text', max_seq_length: Optional[int]=None) -> 'DataPipeline'`</sub> | method | 分词并打包为固定长度序列 | `self._ensure_dataset`, `getattr`, `self._dataset.map`, `self._pack_sequences` |
| [`DataPipeline.tokenize_and_pack._tokenize`](../saddlellm/DataPipeline.py#L395)<br><sub>`_tokenize(examples)`</sub> | nested function | `DataPipeline` 中分词`tokenize`的局部回调/辅助逻辑。 | `tokenizer` |
| [`DataPipeline._pack_sequences`](../saddlellm/DataPipeline.py#L416)<br><sub>`_pack_sequences(self, dataset, max_len: int, pad_token_id: int=0)`</sub> | method | 将多个短序列拼接打包到 max_len | `Dataset.from_generator` |
| [`DataPipeline._pack_sequences._pack_generator`](../saddlellm/DataPipeline.py#L420)<br><sub>`_pack_generator()`</sub> | nested function | `DataPipeline` 中实现`pack_generator`的局部回调/辅助逻辑。 | `example.get`, `buffer.extend`, `len` |
| [`DataPipeline.to_iterable_dataset`](../saddlellm/DataPipeline.py#L439)<br><sub>`to_iterable_dataset(self)`</sub> | method | 转换为 HuggingFace IterableDataset (用于流式训练) | — |
| [`DataPipeline.save_to_disk`](../saddlellm/DataPipeline.py#L443)<br><sub>`save_to_disk(self, path: Optional[str]=None)`</sub> | method | 保存处理后的数据到磁盘 | `os.makedirs`, `hasattr`, `self._dataset.save_to_disk`, `enumerate`, `batch.append`, `len`, `pq.write_table`, `pa.Table.from_pylist`, `os.path.join`, `logger.info` |
| [`DataPipeline.split`](../saddlellm/DataPipeline.py#L469)<br><sub>`split(self, ratios: Optional[List[float]]=None, seed: Optional[int]=None)`</sub> | method | 划分训练/验证集 | `logger.warning`, `self._dataset.train_test_split` |
| [`DataPipeline.stats`](../saddlellm/DataPipeline.py#L484)<br><sub>`stats(self) -> Dict`</sub> | method | `DataPipeline` 中实现`stats`的公开操作。 | `dict` |
| [`DataPipeline._detect_format`](../saddlellm/DataPipeline.py#L487)<br><sub>`_detect_format(self, path: str) -> str`</sub> | method | `DataPipeline` 中检测`detect_format`的内部辅助逻辑。 | `lower`, `os.path.splitext`, `os.path.isdir` |
| [`DataMixer.__init__`](../saddlellm/DataPipeline.py#L568)<br><sub>`__init__(self, configs: List[DataMixConfig], mix_strategy: str='interleave', shuffle_seed: int=42)`</sub> | method | 初始化 `DataMixer` 实例及其运行依赖。 | `sum` |
| [`DataMixer.from_preset`](../saddlellm/DataPipeline.py#L582)<br><sub>`from_preset(cls, preset_name: str, source_paths: Dict[str, List[Dict]]) -> 'DataMixer'`</sub> | method | 从预设配比创建 DataMixer | `join`, `cls.PRESETS.keys`, `KeyError`, `source_paths.get`, `configs.append`, `DataMixConfig`, `cls` |
| [`DataMixer.mix_and_process`](../saddlellm/DataPipeline.py#L597)<br><sub>`mix_and_process(self, tokenizer, max_seq_length: int=2048, num_proc: int=4)`</sub> | method | 混合多源数据并进行处理,返回可用于训练的 dataset | `logger.info`, `PipelineConfig`, `DataPipeline`, `filter_quality`, `deduplicate`, `clean`, `pipeline.collect`, `pipeline.tokenize_and_pack`, `pipeline.to_iterable_dataset`, `concatenate_datasets` |
| [`DataMixer.print_mix_summary`](../saddlellm/DataPipeline.py#L662)<br><sub>`print_mix_summary(self) -> str`</sub> | method | 打印混合配比摘要 | `sum`, `max`, `int`, `lines.append`, `join` |

## `saddlellm/DataStrategy.py`

共 19 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DataBucketPlan.to_dict`](../saddlellm/DataStrategy.py#L22)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `DataBucketPlan` 转为可序列化字典。 | `asdict` |
| [`DataMixPlan.normalized`](../saddlellm/DataStrategy.py#L33)<br><sub>`normalized(self) -> 'DataMixPlan'`</sub> | method | `DataMixPlan` 中实现`normalized`的公开操作。 | `sum`, `max`, `DataMixPlan`, `DataBucketPlan`, `list` |
| [`DataMixPlan.to_dict`](../saddlellm/DataStrategy.py#L54)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `DataMixPlan` 转为可序列化字典。 | `self.normalized`, `b.to_dict` |
| [`DataMixPlan.source_weights`](../saddlellm/DataStrategy.py#L63)<br><sub>`source_weights(self) -> List[float]`</sub> | method | `DataMixPlan` 中实现数据源的公开操作。 | `self.normalized` |
| [`DataMixPlan.to_pipeline_config`](../saddlellm/DataStrategy.py#L66)<br><sub>`to_pipeline_config(self, sources_by_bucket: Dict[str, Sequence[Dict]], output_dir: str='./processed_data', max_seq_length: int=2048, dedup_method: str='minhash', quality_min_score: Optional[float]=None) -> Dict`</sub> | method | `DataMixPlan` 中实现配置的公开操作。 | `self.normalized`, `list`, `sources_by_bucket.get`, `missing.append`, `dict`, `src.setdefault`, `sources.append`, `weights.append`, `len`, `ValueError` |
| [`DataMixPlanner.for_domain`](../saddlellm/DataStrategy.py#L127)<br><sub>`for_domain(cls, domain: str, domain_boost: float=0.0, include_code: bool=True, include_dialogue: bool=True) -> DataMixPlan`</sub> | method | `DataMixPlanner` 中实现`for_domain`的公开操作。 | `lower`, `cls.BASE_BUCKETS.items`, `weights.update`, `cls.DOMAIN_OVERRIDES.get`, `max`, `weights.get`, `DataBucketPlan`, `weights.items`, `warnings.append`, `normalized` |
| [`ContaminationMatch.to_dict`](../saddlellm/DataStrategy.py#L176)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ContaminationMatch` 转为可序列化字典。 | `asdict` |
| [`ContaminationReport.to_dict`](../saddlellm/DataStrategy.py#L188)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ContaminationReport` 转为可序列化字典。 | `m.to_dict` |
| [`ContaminationDetector.__init__`](../saddlellm/DataStrategy.py#L201)<br><sub>`__init__(self, ngram: int=8, threshold: float=0.35)`</sub> | method | 初始化 `ContaminationDetector` 实例及其运行依赖。 | `ValueError` |
| [`ContaminationDetector.add_references`](../saddlellm/DataStrategy.py#L208)<br><sub>`add_references(self, references: Union[Dict[str, str], Sequence[str]])`</sub> | method | `ContaminationDetector` 中添加`add_references`的公开操作。 | `isinstance`, `references.items`, `enumerate`, `self._shingles`, `str` |
| [`ContaminationDetector.add_reference_files`](../saddlellm/DataStrategy.py#L219)<br><sub>`add_reference_files(self, paths: Union[str, Sequence[str]], text_column: str='text', max_samples: Optional[int]=None)`</sub> | method | `ContaminationDetector` 中添加`add_reference_files`的公开操作。 | `enumerate`, `self._iter_files`, `self.add_references` |
| [`ContaminationDetector.scan_texts`](../saddlellm/DataStrategy.py#L231)<br><sub>`scan_texts(self, texts: Sequence[str], threshold: Optional[float]=None) -> ContaminationReport`</sub> | method | `ContaminationDetector` 中实现`scan_texts`的公开操作。 | `self._scan`, `enumerate` |
| [`ContaminationDetector.scan_files`](../saddlellm/DataStrategy.py#L234)<br><sub>`scan_files(self, paths: Union[str, Sequence[str]], text_column: str='text', max_samples: Optional[int]=None, threshold: Optional[float]=None) -> ContaminationReport`</sub> | method | `ContaminationDetector` 中实现`scan_files`的公开操作。 | `self._scan`, `iterator` |
| [`ContaminationDetector.scan_files.iterator`](../saddlellm/DataStrategy.py#L241)<br><sub>`iterator()`</sub> | nested function | `ContaminationDetector` 中实现`iterator`的局部回调/辅助逻辑。 | `enumerate`, `self._iter_files` |
| [`ContaminationDetector.save_report`](../saddlellm/DataStrategy.py#L248)<br><sub>`save_report(self, path: str, report: ContaminationReport) -> str`</sub> | method | `ContaminationDetector` 中保存报告的公开操作。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump`, `report.to_dict` |
| [`ContaminationDetector._scan`](../saddlellm/DataStrategy.py#L254)<br><sub>`_scan(self, samples: Iterator[Tuple[str, str]], threshold: Optional[float]) -> ContaminationReport`</sub> | method | `ContaminationDetector` 中实现`scan`的内部辅助逻辑。 | `ValueError`, `self._shingles`, `self._references.items`, `len`, `min`, `max`, `ContaminationMatch`, `matches.append`, `ContaminationReport` |
| [`ContaminationDetector._shingles`](../saddlellm/DataStrategy.py#L284)<br><sub>`_shingles(self, text: str) -> set`</sub> | method | `ContaminationDetector` 中实现`shingles`的内部辅助逻辑。 | `self._tokens`, `len`, `set`, `join`, `range` |
| [`ContaminationDetector._tokens`](../saddlellm/DataStrategy.py#L290)<br><sub>`_tokens(self, text: str) -> List[str]`</sub> | method | `ContaminationDetector` 中实现`tokens`的内部辅助逻辑。 | `lower`, `re.findall` |
| [`ContaminationDetector._iter_files`](../saddlellm/DataStrategy.py#L294)<br><sub>`_iter_files(self, paths: Union[str, Sequence[str]], text_column: str='text') -> Iterator[str]`</sub> | method | `ContaminationDetector` 中实现`iter_files`的内部辅助逻辑。 | `isinstance`, `list`, `any`, `files.extend`, `glob.glob`, `os.path.isdir`, `os.path.join`, `os.path.isfile`, `files.append`, `sorted` |

## `saddlellm/DatasetManifest.py`

共 15 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DatasetManifestEntry.to_dict`](../saddlellm/DatasetManifest.py#L29)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `DatasetManifestEntry` 转为可序列化字典。 | `asdict` |
| [`DatasetManifestEntry.text_column`](../saddlellm/DatasetManifest.py#L32)<br><sub>`text_column(self) -> str`</sub> | method | `DatasetManifestEntry` 中实现`text_column`的公开操作。 | — |
| [`DatasetManifestEntry.to_saddle_source`](../saddlellm/DatasetManifest.py#L38)<br><sub>`to_saddle_source(self) -> Dict`</sub> | method | Convert to a DataPipeline-compatible source dictionary. | `dict`, `self.text_column`, `source.update` |
| [`DatasetManifest.to_dict`](../saddlellm/DatasetManifest.py#L66)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `DatasetManifest` 转为可序列化字典。 | `entry.to_dict` |
| [`DatasetManifest.from_dict`](../saddlellm/DatasetManifest.py#L75)<br><sub>`from_dict(cls, data: Dict) -> 'DatasetManifest'`</sub> | method | 从字典解析并创建 `DatasetManifest`。 | `DatasetManifestEntry`, `data.get`, `cls` |
| [`DatasetManifest.from_entries`](../saddlellm/DatasetManifest.py#L88)<br><sub>`from_entries(cls, entries: Iterable[DatasetManifestEntry], name: str='saddlellm-dataset-manifest', metadata: Optional[Dict]=None) -> 'DatasetManifest'`</sub> | method | `DatasetManifest` 中实现`from_entries`的公开操作。 | `cls`, `list` |
| [`DatasetManifest.load`](../saddlellm/DatasetManifest.py#L97)<br><sub>`load(cls, path: str) -> 'DatasetManifest'`</sub> | method | `DatasetManifest` 中加载`load`的公开操作。 | `open`, `cls.from_dict`, `json.load` |
| [`DatasetManifest.save`](../saddlellm/DatasetManifest.py#L101)<br><sub>`save(self, path: str) -> str`</sub> | method | `DatasetManifest` 中保存`save`的公开操作。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump`, `self.to_dict` |
| [`DatasetManifest.from_llamafactory_json`](../saddlellm/DatasetManifest.py#L108)<br><sub>`from_llamafactory_json(cls, path: str, dataset_dir: Optional[str]=None, selected_names: Optional[Sequence[str]]=None, default_role: str='sft', default_weight: float=1.0) -> 'DatasetManifest'`</sub> | method | Load a LLaMA-Factory-style dataset_info.json into a neutral manifest. | `open`, `json.load`, `set`, `raw.keys`, `os.path.dirname`, `raw.items`, `cls._resolve_llamafactory_source`, `cls._infer_role`, `DatasetManifestEntry`, `info.get` |
| [`DatasetManifest.filter`](../saddlellm/DatasetManifest.py#L175)<br><sub>`filter(self, role: Optional[str]=None, formatting: Optional[str]=None, ranking: Optional[bool]=None) -> 'DatasetManifest'`</sub> | method | `DatasetManifest` 中过滤`filter`的公开操作。 | `DatasetManifest`, `dict` |
| [`DatasetManifest.to_saddle_sources`](../saddlellm/DatasetManifest.py#L190)<br><sub>`to_saddle_sources(self, role: Optional[str]=None) -> List[Dict]`</sub> | method | `DatasetManifest` 中实现`to_saddle_sources`的公开操作。 | `self.filter`, `entry.to_saddle_source` |
| [`DatasetManifest.source_weights`](../saddlellm/DatasetManifest.py#L194)<br><sub>`source_weights(self, role: Optional[str]=None, normalize: bool=True) -> List[float]`</sub> | method | `DatasetManifest` 中实现数据源的公开操作。 | `self.filter`, `max`, `float`, `sum` |
| [`DatasetManifest.summary`](../saddlellm/DatasetManifest.py#L202)<br><sub>`summary(self) -> Dict`</sub> | method | `DatasetManifest` 中实现`summary`的公开操作。 | `by_role.get`, `by_formatting.get`, `by_source_type.get`, `len`, `sum`, `float` |
| [`DatasetManifest._resolve_llamafactory_source`](../saddlellm/DatasetManifest.py#L221)<br><sub>`_resolve_llamafactory_source(info: Dict, base_dir: str) -> (str, str)`</sub> | method | `DatasetManifest` 中解析数据源的内部辅助逻辑。 | `os.path.abspath`, `os.path.join` |
| [`DatasetManifest._infer_role`](../saddlellm/DatasetManifest.py#L237)<br><sub>`_infer_role(info: Dict, default_role: str) -> str`</sub> | method | `DatasetManifest` 中推断`infer_role`的内部辅助逻辑。 | `info.get` |

## `saddlellm/DensePretrainer.py`

共 19 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`init_weights_llama_style`](../saddlellm/DensePretrainer.py#L30)<br><sub>`init_weights_llama_style(model: nn.Module, initializer_range: float=0.02)`</sub> | function | Llama 风格的初始化 —— 对训练 from-scratch 的稳定性至关重要。 | `model.named_parameters`, `param.dim`, `nn.init.normal_`, `math.sqrt`, `nn.init.constant_`, `nn.init.zeros_`, `logger.info` |
| [`init_weights_small_embed`](../saddlellm/DensePretrainer.py#L62)<br><sub>`init_weights_small_embed(model: nn.Module, initializer_range: float=0.02)`</sub> | function | 对小模型的深度缩放初始化。 | `model.named_parameters`, `param.dim`, `nn.init.normal_`, `nn.init.constant_`, `nn.init.zeros_`, `logger.info` |
| [`init_weights_deepseek_style`](../saddlellm/DensePretrainer.py#L87)<br><sub>`init_weights_deepseek_style(model: nn.Module)`</sub> | function | DeepSeek 风格初始化。 | `model.named_parameters`, `param.dim`, `nn.init.normal_`, `math.sqrt`, `nn.init.zeros_`, `logger.info` |
| [`DensePretrainer.__init__`](../saddlellm/DensePretrainer.py#L214)<br><sub>`__init__(self, model: nn.Module, tokenizer, train_dataset, config: DensePretrainConfig=None, val_dataset=None)`</sub> | method | 初始化 `DensePretrainer` 实例及其运行依赖。 | `DensePretrainConfig`, `next`, `model.parameters`, `float`, `self._detect_flash_attention`, `self._set_seed`, `torch.cuda.is_available` |
| [`DensePretrainer.initialize_model`](../saddlellm/DensePretrainer.py#L258)<br><sub>`initialize_model(self, style: str='llama')`</sub> | method | 初始化模型权重 (必须在移动到 GPU 之后调用)。 | `init_weights_llama_style`, `init_weights_small_embed`, `init_weights_deepseek_style`, `logger.warning` |
| [`DensePretrainer.train`](../saddlellm/DensePretrainer.py#L276)<br><sub>`train(self)`</sub> | method | 主训练循环 | `logger.info`, `sum`, `p.numel`, `self.model.parameters`, `self._setup_optimizer`, `self._setup_dataloader`, `torch.cuda.amp.GradScaler`, `self.model.gradient_checkpointing_enable`, `torch.compile`, `logger.warning` |
| [`DensePretrainer._setup_optimizer`](../saddlellm/DensePretrainer.py#L430)<br><sub>`_setup_optimizer(self)`</sub> | method | 配置 AdamW + Cosine LR Schedule | `self.model.named_parameters`, `any`, `no_decay_params.append`, `decay_params.append`, `torch.optim.AdamW`, `torch.optim.lr_scheduler.LambdaLR`, `logger.info` |
| [`DensePretrainer._setup_optimizer.lr_lambda`](../saddlellm/DensePretrainer.py#L458)<br><sub>`lr_lambda(step)`</sub> | nested function | `DensePretrainer` 中实现`lr_lambda`的局部回调/辅助逻辑。 | `max`, `math.cos` |
| [`DensePretrainer._setup_dataloader`](../saddlellm/DensePretrainer.py#L469)<br><sub>`_setup_dataloader(self)`</sub> | method | 设置可复现的 DataLoader | `platform.system`, `DataLoader`, `min` |
| [`DensePretrainer._setup_dataloader.collate`](../saddlellm/DensePretrainer.py#L477)<br><sub>`collate(batch)`</sub> | nested function | `DensePretrainer` 中实现`collate`的局部回调/辅助逻辑。 | `isinstance`, `keys`, `torch.stack`, `torch.tensor` |
| [`DensePretrainer._detect_loss_spike`](../saddlellm/DensePretrainer.py#L497)<br><sub>`_detect_loss_spike(self, loss_value: float) -> bool`</sub> | method | 检测 loss spike: loss > moving_avg * threshold | `len` |
| [`DensePretrainer._update_loss_moving_avg`](../saddlellm/DensePretrainer.py#L505)<br><sub>`_update_loss_moving_avg(self)`</sub> | method | `DensePretrainer` 中更新`update_loss_moving_avg`的内部辅助逻辑。 | `sum`, `len` |
| [`DensePretrainer._detect_flash_attention`](../saddlellm/DensePretrainer.py#L511)<br><sub>`_detect_flash_attention(self) -> bool`</sub> | method | 检测 Flash Attention 是否可用 | `importlib.util.find_spec`, `logger.info`, `hasattr` |
| [`DensePretrainer._run_validation`](../saddlellm/DensePretrainer.py#L534)<br><sub>`_run_validation(self, step: int)`</sub> | method | 运行验证: 计算 perplexity | `self.model.eval`, `torch.no_grad`, `enumerate`, `v.to`, `batch.items`, `self.model`, `outputs.loss.item`, `numel`, `math.exp`, `max` |
| [`DensePretrainer._log_step`](../saddlellm/DensePretrainer.py#L556)<br><sub>`_log_step(self, step: int, loss: float, grad_norm, lr: float, mfu: Dict)`</sub> | method | 单行日志 | `mfu.get`, `parts.append`, `logger.info`, `join` |
| [`DensePretrainer._save_checkpoint`](../saddlellm/DensePretrainer.py#L569)<br><sub>`_save_checkpoint(self, step)`</sub> | method | 保存检查点 | `os.path.join`, `os.makedirs`, `self.model.save_pretrained`, `self.tokenizer.save_pretrained`, `self.optimizer.state_dict`, `self.scheduler.state_dict`, `torch.save`, `logger.info` |
| [`DensePretrainer._load_checkpoint`](../saddlellm/DensePretrainer.py#L591)<br><sub>`_load_checkpoint(self, checkpoint_dir: str=None) -> bool`</sub> | method | 恢复训练状态 | `os.path.join`, `os.path.exists`, `torch.load`, `checkpoint.get`, `float`, `self.optimizer.load_state_dict`, `self.scheduler.load_state_dict`, `logger.info` |
| [`DensePretrainer._set_seed`](../saddlellm/DensePretrainer.py#L608)<br><sub>`_set_seed(self, seed: int)`</sub> | method | `DensePretrainer` 中设置`set_seed`的内部辅助逻辑。 | `random.seed`, `torch.manual_seed`, `torch.cuda.is_available`, `torch.cuda.manual_seed_all` |
| [`DensePretrainer.lr_range_test`](../saddlellm/DensePretrainer.py#L618)<br><sub>`lr_range_test(self, min_lr: float=1e-06, max_lr: float=0.01, steps: int=500, smooth_window: int=20) -> Dict`</sub> | method | 学习率范围测试 (Leslie Smith 方法)。 | `self.model.train`, `torch.optim.AdamW`, `self.model.parameters`, `range`, `next`, `iter`, `self._setup_dataloader`, `v.to`, `batch.items`, `self.model` |

## `saddlellm/Deploy.py`

共 8 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`Deploy.export`](../saddlellm/Deploy.py#L38)<br><sub>`export(model, tokenizer, output_dir: str='./exported_model', format: str='hf', quantize: bool=False, quantize_bits: int=4)`</sub> | method | 导出模型。 | `os.makedirs`, `Deploy._export_hf`, `Deploy._export_onnx`, `Deploy._export_gguf`, `ValueError`, `sum`, `p.numel`, `model.parameters`, `open`, `os.path.join` |
| [`Deploy._export_hf`](../saddlellm/Deploy.py#L81)<br><sub>`_export_hf(model, tokenizer, output_dir, quantize, bits)`</sub> | method | 导出 HuggingFace 格式。 | `ModelQuantizer`, `q.quantize`, `logger.warning`, `model.save_pretrained`, `tokenizer.save_pretrained`, `logger.info` |
| [`Deploy._export_onnx`](../saddlellm/Deploy.py#L96)<br><sub>`_export_onnx(model, tokenizer, output_dir)`</sub> | method | 导出 ONNX 格式。 | `model.eval`, `tokenizer`, `os.path.join`, `torch.onnx.export`, `tokenizer.save_pretrained`, `logger.info`, `logger.warning`, `Deploy._export_hf` |
| [`Deploy._export_gguf`](../saddlellm/Deploy.py#L123)<br><sub>`_export_gguf(model, tokenizer, output_dir)`</sub> | method | 导出 GGUF 格式 (需要 llama-cpp-python)。 | `os.path.join`, `Deploy._export_hf`, `subprocess.run`, `logger.info`, `logger.warning`, `os.path.exists`, `shutil.rmtree` |
| [`Deploy.serve`](../saddlellm/Deploy.py#L156)<br><sub>`serve(model, tokenizer, host: str='0.0.0.0', port: int=8000, title: str='SaddleLLM API', max_tokens: int=2048)`</sub> | method | 启动 FastAPI 推理服务。 | `ImportError`, `next`, `model.parameters`, `FastAPI`, `print`, `uvicorn.run` |
| [`Deploy.serve.health`](../saddlellm/Deploy.py#L223)<br><sub>`async health()`</sub> | nested function | `Deploy` 中实现`health`的局部回调/辅助逻辑。 | — |
| [`Deploy.serve.chat_completions`](../saddlellm/Deploy.py#L227)<br><sub>`async chat_completions(req: ChatRequest)`</sub> | nested function | `Deploy` 中实现`chat_completions`的局部回调/辅助逻辑。 | `isinstance`, `model.eval`, `to`, `tokenizer`, `torch.no_grad`, `model.generate`, `tokenizer.decode`, `len`, `strip`, `ChatResponse` |
| [`Deploy.serve.generate`](../saddlellm/Deploy.py#L263)<br><sub>`async generate(req: GenerateRequest)`</sub> | nested function | `Deploy` 中生成`generate`的局部回调/辅助逻辑。 | `model.eval`, `to`, `tokenizer`, `torch.no_grad`, `model.generate`, `tokenizer.decode`, `len`, `strip`, `GenerateResponse` |

## `saddlellm/DistillationTrainer.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DistillationTrainer.__init__`](../saddlellm/DistillationTrainer.py#L27)<br><sub>`__init__(self, teacher: str, student: str, temperature: float=2.0, alpha_ce: float=0.5, alpha_hidden: float=0.3, max_length: int=1024, device_map: str='auto')`</sub> | method | Initialize distillation trainer. | `AutoModelForCausalLM.from_pretrained`, `self.teacher.eval`, `self.teacher.parameters`, `AutoTokenizer.from_pretrained`, `KLDivLoss`, `torch.nn.CrossEntropyLoss` |
| [`DistillationTrainer.compute_distillation_loss`](../saddlellm/DistillationTrainer.py#L77)<br><sub>`compute_distillation_loss(self, student_outputs, teacher_outputs, labels)`</sub> | method | 计算蒸馏损失（logits + 可选隐藏层匹配） | `self.kl_loss`, `torch.nn.functional.log_softmax`, `torch.nn.functional.softmax`, `contiguous`, `self.ce_loss`, `shift_logits.view`, `shift_logits.size`, `shift_labels.view`, `detach`, `torch.nn.functional.mse_loss` |
| [`DistillationTrainer.fit`](../saddlellm/DistillationTrainer.py#L113)<br><sub>`fit(self, train_dataset: Dataset, eval_dataset: Optional[Dataset]=None, epochs: int=3, batch_size: int=2, learning_rate: float=5e-05, output_dir: str='./distill_output', logging_steps: int=10)`</sub> | method | Run distillation training. | `train_dataset.map`, `eval_dataset.map`, `AdamW`, `self.student.parameters`, `DataCollatorForLanguageModeling`, `range`, `self.student.train`, `enumerate`, `to`, `self.tokenizer` |
| [`DistillationTrainer.fit.tokenize_fn`](../saddlellm/DistillationTrainer.py#L125)<br><sub>`tokenize_fn(examples: Dict) -> Dict`</sub> | nested function | `DistillationTrainer` 中分词`tokenize_fn`的局部回调/辅助逻辑。 | `self.tokenizer` |
| [`DistillationTrainer.predict`](../saddlellm/DistillationTrainer.py#L182)<br><sub>`predict(self, text: str, max_new_tokens: int=100) -> str`</sub> | method | 使用学生模型生成文本 | `to`, `self.tokenizer`, `self.student.generate`, `self.tokenizer.decode` |
| [`DistillationTrainer.save`](../saddlellm/DistillationTrainer.py#L198)<br><sub>`save(self, path: str)`</sub> | method | 保存学生模型 | `self.student.save_pretrained`, `self.tokenizer.save_pretrained` |
| [`DistillationTrainer.load`](../saddlellm/DistillationTrainer.py#L204)<br><sub>`load(cls, path: str, **kwargs)`</sub> | method | 加载蒸馏后的学生模型 | `cls` |

## `saddlellm/DistributedConfig.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DistributedConfig.__post_init__`](../saddlellm/DistributedConfig.py#L43)<br><sub>`__post_init__(self)`</sub> | method | 在 `DistributedConfig` 创建后校验并规范化字段。 | `lower`, `str`, `ValueError`, `int` |
| [`DistributedConfig.to_training_args`](../saddlellm/DistributedConfig.py#L55)<br><sub>`to_training_args(self) -> Dict`</sub> | method | 转换为 HuggingFace TrainingArguments 可用的参数 | `torch.cuda.is_available`, `upper`, `str`, `self._fsdp_config_dict`, `importlib.util.find_spec`, `RuntimeError`, `self._deepspeed_config_dict` |
| [`DistributedConfig.to_deepspeed_config`](../saddlellm/DistributedConfig.py#L85)<br><sub>`to_deepspeed_config(self) -> Dict`</sub> | method | 生成 DeepSpeed 完整配置 | — |
| [`DistributedConfig.save_deepspeed_config`](../saddlellm/DistributedConfig.py#L135)<br><sub>`save_deepspeed_config(self, path: str) -> str`</sub> | method | 保存 DeepSpeed 配置到 JSON 文件 | `self.to_deepspeed_config`, `os.makedirs`, `os.path.dirname`, `open`, `json.dump` |
| [`DistributedConfig.save_accelerate_config`](../saddlellm/DistributedConfig.py#L143)<br><sub>`save_accelerate_config(self, path: str) -> str`</sub> | method | Save a minimal Accelerate config for the selected strategy. | `max`, `self.to_deepspeed_config`, `self._fsdp_config_dict`, `os.makedirs`, `os.path.dirname`, `open`, `json.dump` |
| [`DistributedConfig.recommend_for_model`](../saddlellm/DistributedConfig.py#L173)<br><sub>`recommend_for_model(cls, params: int, gpu_memory_gb: float, num_gpus: int=1, prefer_offload: bool=False) -> 'DistributedConfig'`</sub> | method | Choose a conservative distributed strategy from model size and GPU memory. | `max`, `cls` |
| [`DistributedConfig.save_template_bundle`](../saddlellm/DistributedConfig.py#L200)<br><sub>`save_template_bundle(cls, output_dir: str, params: int, gpu_memory_gb: float, num_gpus: int=1) -> Dict[str, str]`</sub> | method | Create DeepSpeed and Accelerate templates for a planned pretraining run. | `os.makedirs`, `cls.recommend_for_model`, `cfg.save_accelerate_config`, `os.path.join`, `cfg.save_deepspeed_config` |
| [`DistributedConfig._fsdp_config_dict`](../saddlellm/DistributedConfig.py#L218)<br><sub>`_fsdp_config_dict(self) -> Dict`</sub> | method | `DistributedConfig` 中实现配置的内部辅助逻辑。 | `upper`, `str` |
| [`DistributedConfig._deepspeed_config_dict`](../saddlellm/DistributedConfig.py#L230)<br><sub>`_deepspeed_config_dict(self) -> Dict`</sub> | method | `DistributedConfig` 中实现配置的内部辅助逻辑。 | `self.to_deepspeed_config` |
| [`DistributedConfig.auto_detect_gpus`](../saddlellm/DistributedConfig.py#L234)<br><sub>`auto_detect_gpus() -> int`</sub> | method | 自动检测可用 GPU 数量 | `torch.cuda.is_available`, `torch.cuda.device_count` |
| [`DistributedConfig.recommend_strategy`](../saddlellm/DistributedConfig.py#L242)<br><sub>`recommend_strategy(num_gpus: int) -> str`</sub> | method | 根据 GPU 数量推荐策略 | — |

## `saddlellm/DistributedRuntime.py`

共 8 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DistributedRuntimeInfo.is_distributed`](../saddlellm/DistributedRuntime.py#L20)<br><sub>`is_distributed(self) -> bool`</sub> | method | `DistributedRuntimeInfo` 中实现分布式运行时的公开操作。 | — |
| [`DistributedRuntimeInfo.is_main_process`](../saddlellm/DistributedRuntime.py#L24)<br><sub>`is_main_process(self) -> bool`</sub> | method | `DistributedRuntimeInfo` 中处理`is_main_process`的公开操作。 | — |
| [`DistributedRuntimeInfo.to_dict`](../saddlellm/DistributedRuntime.py#L27)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `DistributedRuntimeInfo` 转为可序列化字典。 | `asdict` |
| [`DistributedRuntimeInfo.from_env`](../saddlellm/DistributedRuntime.py#L31)<br><sub>`from_env(cls, strategy: str='single') -> 'DistributedRuntimeInfo'`</sub> | method | `DistributedRuntimeInfo` 中实现`from_env`的公开操作。 | `cls`, `str`, `_env_int` |
| [`current_distributed_runtime`](../saddlellm/DistributedRuntime.py#L41)<br><sub>`current_distributed_runtime(strategy: str='single') -> DistributedRuntimeInfo`</sub> | function | Return initialized torch.distributed state, falling back to launcher env. | `DistributedRuntimeInfo.from_env`, `dist.is_available`, `dist.is_initialized`, `DistributedRuntimeInfo`, `str`, `int`, `dist.get_rank`, `dist.get_world_size` |
| [`validate_distributed_runtime`](../saddlellm/DistributedRuntime.py#L61)<br><sub>`validate_distributed_runtime(strategy: str, num_gpus: int, num_nodes: int=1, *, require_initialized: bool=False) -> DistributedRuntimeInfo`</sub> | function | Fail when a distributed config is executed by an incompatible launcher. | `lower`, `str`, `current_distributed_runtime`, `int`, `ValueError`, `RuntimeError` |
| [`distributed_barrier`](../saddlellm/DistributedRuntime.py#L101)<br><sub>`distributed_barrier() -> None`</sub> | function | 模块级实现分布式运行时的公开操作。 | `dist.is_available`, `dist.is_initialized`, `dist.get_world_size`, `dist.barrier` |
| [`_env_int`](../saddlellm/DistributedRuntime.py#L111)<br><sub>`_env_int(name: str, default: int) -> int`</sub> | function | 模块级实现`env_int`的内部辅助逻辑。 | `os.environ.get`, `int`, `ValueError` |

## `saddlellm/DomainBuilder.py`

共 19 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DomainModelBuilder.__init__`](../saddlellm/DomainBuilder.py#L78)<br><sub>`__init__(self, domain: str, base_model: str, output_dir: str='./domain_model', model_config: str='llama-300m', tokenizer_name_or_path: Optional[str]=None)`</sub> | method | 初始化 `DomainModelBuilder` 实例及其运行依赖。 | `domain.lower`, `join`, `sorted`, `KeyError`, `DomainBuildConfig` |
| [`DomainModelBuilder.architecture_report`](../saddlellm/DomainBuilder.py#L99)<br><sub>`architecture_report(self) -> Dict`</sub> | method | `DomainModelBuilder` 中报告报告的公开操作。 | `ArchitectureRegistry.detect_from_model_name`, `support.to_dict`, `asdict` |
| [`DomainModelBuilder.training_strategy`](../saddlellm/DomainBuilder.py#L109)<br><sub>`training_strategy(self, target_family: Optional[str]=None, budget: str='low', prefer_scratch: bool=False, teacher_models: Optional[Sequence[str]]=None) -> Dict`</sub> | method | `DomainModelBuilder` 中实现训练的公开操作。 | `to_dict`, `TrainingStrategyAdvisor.analyze` |
| [`DomainModelBuilder.low_cost_recipe`](../saddlellm/DomainBuilder.py#L128)<br><sub>`low_cost_recipe(self, target_family: Optional[str]=None, teacher_models: Optional[Sequence[str]]=None) -> List[str]`</sub> | method | `DomainModelBuilder` 中实现训练配方的公开操作。 | `TrainingStrategyAdvisor.low_cost_recipe` |
| [`DomainModelBuilder.data_mix_plan`](../saddlellm/DomainBuilder.py#L142)<br><sub>`data_mix_plan(self, sources_by_bucket: Optional[Dict[str, Sequence[Dict]]]=None, output_dir: Optional[str]=None, max_seq_length: Optional[int]=None, domain_boost: float=0.0) -> Dict`</sub> | method | `DomainModelBuilder` 中规划数据、计划的公开操作。 | `DataMixPlanner.for_domain`, `plan.to_pipeline_config`, `os.path.join`, `plan.to_dict` |
| [`DomainModelBuilder.contamination_report`](../saddlellm/DomainBuilder.py#L160)<br><sub>`contamination_report(self, reference_texts: Sequence[str], corpus_files: Union[str, Sequence[str]], text_column: str='text', threshold: float=0.35, max_samples: Optional[int]=None) -> Dict`</sub> | method | `DomainModelBuilder` 中报告报告的公开操作。 | `add_references`, `ContaminationDetector`, `to_dict`, `detector.scan_files` |
| [`DomainModelBuilder.eval_suite`](../saddlellm/DomainBuilder.py#L177)<br><sub>`eval_suite(self, max_samples: int=500, save_path: Optional[str]=None) -> Dict`</sub> | method | `DomainModelBuilder` 中实现`eval_suite`的公开操作。 | `PretrainEvalSuite.for_domain`, `suite.save`, `suite.to_dict` |
| [`DomainModelBuilder.pretrain_experiments`](../saddlellm/DomainBuilder.py#L185)<br><sub>`pretrain_experiments(self, output_dir: Optional[str]=None, model_names: Optional[List[str]]=None, token_multipliers: Optional[List[float]]=None, corpus_sources: Optional[Sequence[Dict]]=None, sources_by_bucket: Optional[Dict[str, Sequence[Dict]]]=None, dry_run: bool=True) -> Dict`</sub> | method | `DomainModelBuilder` 中实现`pretrain_experiments`的公开操作。 | `PretrainExperimentRunner`, `PretrainExperimentConfig`, `os.path.join`, `list`, `to_dict`, `runner.run` |
| [`DomainModelBuilder.build_config`](../saddlellm/DomainBuilder.py#L208)<br><sub>`build_config(self, corpus_sources: Optional[Sequence[Union[str, Dict]]]=None, sft_data: Optional[str]=None, preference_data: Optional[str]=None, eval_data: Optional[str]=None, stages: Optional[List[str]]=None, max_steps: int=1000, sft_epochs: int=1, preference_epochs: int=1, learning_rate: float=2e-05, sft_learning_rate: float=0.0002, preference_learning_rate: float=5e-06) -> Dict`</sub> | method | `DomainModelBuilder` 中构建配置的公开操作。 | `self._default_stages`, `self._normalize_sources`, `self._safe_model_name`, `os.path.join`, `min`, `max`, `bool`, `asdict` |
| [`DomainModelBuilder.save_config`](../saddlellm/DomainBuilder.py#L289)<br><sub>`save_config(self, path: str, config: Dict) -> str`</sub> | method | `DomainModelBuilder` 中保存配置的公开操作。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump` |
| [`DomainModelBuilder.create_orchestrator`](../saddlellm/DomainBuilder.py#L295)<br><sub>`create_orchestrator(self, config: Dict=None)`</sub> | method | `DomainModelBuilder` 中创建`create_orchestrator`的公开操作。 | `TrainingOrchestrator.from_dict`, `self.build_config` |
| [`DomainModelBuilder.run`](../saddlellm/DomainBuilder.py#L300)<br><sub>`run(self, config: Dict=None)`</sub> | method | `DomainModelBuilder` 中执行`run`的公开操作。 | `self.create_orchestrator`, `orchestrator.run` |
| [`DomainModelBuilder.sft`](../saddlellm/DomainBuilder.py#L305)<br><sub>`sft(self, dataset_path: str, output_path: Optional[str]=None, **kwargs)`</sub> | method | `DomainModelBuilder` 中实现`sft`的公开操作。 | `train_model`, `kwargs.pop`, `os.path.join` |
| [`DomainModelBuilder.preference_train`](../saddlellm/DomainBuilder.py#L315)<br><sub>`preference_train(self, dataset_path: str, output_path: Optional[str]=None, method: str='dpo', **kwargs)`</sub> | method | `DomainModelBuilder` 中训练`preference_train`的公开操作。 | `PreferenceTrainConfig`, `kwargs.pop`, `os.path.join`, `train_preference` |
| [`DomainModelBuilder.evaluate`](../saddlellm/DomainBuilder.py#L327)<br><sub>`evaluate(self, dataset_path: str, model_path: Optional[str]=None, **kwargs)`</sub> | method | `DomainModelBuilder` 中评估`evaluate`的公开操作。 | `evaluate`, `Evaluator`, `kwargs.pop` |
| [`DomainModelBuilder._default_stages`](../saddlellm/DomainBuilder.py#L337)<br><sub>`_default_stages(self, corpus_sources, sft_data, preference_data, eval_data) -> List[str]`</sub> | method | `DomainModelBuilder` 中实现`default_stages`的内部辅助逻辑。 | `stages.append` |
| [`DomainModelBuilder._normalize_sources`](../saddlellm/DomainBuilder.py#L349)<br><sub>`_normalize_sources(self, sources: Sequence[Union[str, Dict]]) -> List[Dict]`</sub> | method | `DomainModelBuilder` 中规范化`normalize_sources`的内部辅助逻辑。 | `isinstance`, `normalized.append`, `dict` |
| [`DomainModelBuilder._safe_model_name`](../saddlellm/DomainBuilder.py#L364)<br><sub>`_safe_model_name(self, model_name: str) -> str`</sub> | method | `DomainModelBuilder` 中实现模型的内部辅助逻辑。 | `replace`, `model_name.replace` |
| [`list_domain_recipes`](../saddlellm/DomainBuilder.py#L368)<br><sub>`list_domain_recipes() -> Dict[str, Dict]`</sub> | function | 模块级列出`list_domain_recipes`的公开操作。 | `asdict`, `DOMAIN_RECIPES.items` |

## `saddlellm/ExclusiveTechniques.py`

共 48 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MultiTokenPredictionHead.__init__`](../saddlellm/ExclusiveTechniques.py#L59)<br><sub>`__init__(self, hidden_size: int, vocab_size: int, num_extra_tokens: int=3, shared_head: bool=True)`</sub> | method | 初始化 `MultiTokenPredictionHead` 实例及其运行依赖。 | `__init__`, `super`, `nn.ModuleList`, `nn.Linear`, `range`, `nn.LayerNorm` |
| [`MultiTokenPredictionHead.forward`](../saddlellm/ExclusiveTechniques.py#L92)<br><sub>`forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]`</sub> | method | Args: hidden_states: (batch, seq_len, hidden_size) | `range`, `self.extra_block`, `outputs.append` |
| [`MultiTokenPredictionLoss.__init__`](../saddlellm/ExclusiveTechniques.py#L129)<br><sub>`__init__(self, weights: Optional[List[float]]=None)`</sub> | method | 初始化 `MultiTokenPredictionLoss` 实例及其运行依赖。 | — |
| [`MultiTokenPredictionLoss.compute`](../saddlellm/ExclusiveTechniques.py#L132)<br><sub>`compute(self, logits_list: List[torch.Tensor], labels: torch.Tensor, loss_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Dict]`</sub> | method | 计算 MTP loss。 | `enumerate`, `len`, `shifted_logits.reshape`, `shifted_labels.reshape`, `F.cross_entropy`, `head_loss.item` |
| [`add_mtp_to_model`](../saddlellm/ExclusiveTechniques.py#L174)<br><sub>`add_mtp_to_model(model: nn.Module, vocab_size: int=None, num_extra_tokens: int=3)`</sub> | function | 给现有模型添加 MTP head。 | `model.named_parameters`, `getattr`, `MultiTokenPredictionHead`, `MultiTokenPredictionLoss`, `logger.info` |
| [`AuxLossFreeRouter.__init__`](../saddlellm/ExclusiveTechniques.py#L234)<br><sub>`__init__(self, num_experts: int, hidden_size: int, bias_update_rate: float=0.001, target_load: float=1.0)`</sub> | method | 初始化 `AuxLossFreeRouter` 实例及其运行依赖。 | `__init__`, `super`, `self.register_buffer`, `torch.zeros` |
| [`AuxLossFreeRouter.forward`](../saddlellm/ExclusiveTechniques.py#L253)<br><sub>`forward(self, router_logits: torch.Tensor) -> torch.Tensor`</sub> | method | 在 router logits 上加上 bias, 引导均衡路由。 | `unsqueeze`, `self.expert_bias.unsqueeze` |
| [`AuxLossFreeRouter.update_bias`](../saddlellm/ExclusiveTechniques.py#L265)<br><sub>`update_bias(self, expert_indices: torch.Tensor)`</sub> | method | 根据本轮负载更新 bias。 | `torch.no_grad`, `float`, `torch.bincount`, `expert_indices.flatten`, `load.mean`, `expert_indices.numel` |
| [`AuxLossFreeRouter.get_load_balance`](../saddlellm/ExclusiveTechniques.py#L292)<br><sub>`get_load_balance(self) -> float`</sub> | method | 返回负载均衡指标 (0-1, 1=完美均衡)。 | `self._total_tokens.item`, `sum`, `torch.log`, `math.log`, `item` |
| [`AuxLossFreeMoELayer.__init__`](../saddlellm/ExclusiveTechniques.py#L313)<br><sub>`__init__(self, hidden_size: int, intermediate_size: int, num_experts: int=8, top_k: int=2, bias_update_rate: float=0.001)`</sub> | method | 初始化 `AuxLossFreeMoELayer` 实例及其运行依赖。 | `__init__`, `super`, `nn.Linear`, `AuxLossFreeRouter`, `nn.ModuleList`, `range` |
| [`AuxLossFreeMoELayer.forward`](../saddlellm/ExclusiveTechniques.py#L345)<br><sub>`forward(self, hidden_states: torch.Tensor, update_bias: bool=True)`</sub> | method | Args: hidden_states: (batch, seq_len, hidden_size) update_bias: 是否更新负载均衡 bias (训练时 True, 推理时 False) | `hidden_states.view`, `self.router`, `self.aux_free_router`, `torch.topk`, `F.softmax`, `self.aux_free_router.update_bias`, `torch.zeros_like`, `range`, `any`, `mask.any` |
| [`IterativeAlignment.__init__`](../saddlellm/ExclusiveTechniques.py#L427)<br><sub>`__init__(self, model, ref_model, tokenizer, num_candidates_per_prompt: int=8, keep_best_ratio: float=0.25, quality_threshold: float=0.6)`</sub> | method | 初始化 `IterativeAlignment` 实例及其运行依赖。 | `next`, `model.parameters` |
| [`IterativeAlignment.generate_and_filter`](../saddlellm/ExclusiveTechniques.py#L448)<br><sub>`generate_and_filter(self, prompts: List[str], reward_fn: Optional[Callable]=None) -> List[Dict]`</sub> | method | 生成候选 → 打分 → 筛选, 产生下一轮训练数据。 | `RejectionSampling`, `ProcessRewardModel`, `rs.generate_candidates`, `len`, `prm.split_into_steps`, `prm.score_steps_by_rules`, `prm.aggregate_scores`, `reward_fn`, `scored.append`, `scored.sort` |
| [`IterativeAlignment.train_one_round`](../saddlellm/ExclusiveTechniques.py#L508)<br><sub>`train_one_round(self, preference_data: List[Dict], method: str='dpo', learning_rate: float=5e-06, epochs: int=1, **kwargs)`</sub> | method | 用筛选出的偏好数据训练一轮。 | `logger.info`, `len`, `self._train_dpo`, `self._train_grpo`, `self._train_sft`, `self.ref_model.load_state_dict`, `self.model.state_dict`, `sum`, `d.get`, `max` |
| [`IterativeAlignment._train_dpo`](../saddlellm/ExclusiveTechniques.py#L547)<br><sub>`_train_dpo(self, data, lr, epochs, beta=0.1)`</sub> | method | 简化版 DPO 训练。 | `torch.optim.AdamW`, `self.model.parameters`, `get_scheduler`, `len`, `self.model.train`, `range`, `self._batchify`, `self._dpo_loss`, `loss.backward`, `optimizer.step` |
| [`IterativeAlignment._dpo_loss`](../saddlellm/ExclusiveTechniques.py#L568)<br><sub>`_dpo_loss(self, batch: List[Dict], beta: float) -> torch.Tensor`</sub> | method | 计算 DPO loss。 | `to`, `self.tokenizer`, `torch.no_grad`, `self.ref_model`, `self.model`, `self._seq_log_prob`, `F.logsigmoid`, `len` |
| [`IterativeAlignment._seq_log_prob`](../saddlellm/ExclusiveTechniques.py#L605)<br><sub>`_seq_log_prob(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor`</sub> | method | 序列 log-probability。 | `F.log_softmax`, `sum`, `squeeze`, `torch.gather`, `shift_labels.unsqueeze` |
| [`IterativeAlignment._train_grpo`](../saddlellm/ExclusiveTechniques.py#L612)<br><sub>`_train_grpo(self, data, **kwargs)`</sub> | method | `IterativeAlignment` 中训练`train_grpo`的内部辅助逻辑。 | `GRPOTrainer`, `GRPOConfig`, `trainer.train`, `len` |
| [`IterativeAlignment._train_sft`](../saddlellm/ExclusiveTechniques.py#L618)<br><sub>`_train_sft(self, data, lr, epochs)`</sub> | method | `IterativeAlignment` 中训练`train_sft`的内部辅助逻辑。 | `torch.optim.AdamW`, `self.model.parameters`, `self.model.train`, `range`, `self._batchify`, `to`, `self.tokenizer`, `self.model`, `loss.backward`, `optimizer.step` |
| [`IterativeAlignment._batchify`](../saddlellm/ExclusiveTechniques.py#L636)<br><sub>`_batchify(self, data, batch_size)`</sub> | method | `IterativeAlignment` 中实现`batchify`的内部辅助逻辑。 | `range`, `len` |
| [`IterativeAlignment.save`](../saddlellm/ExclusiveTechniques.py#L640)<br><sub>`save(self, path: str)`</sub> | method | `IterativeAlignment` 中保存`save`的公开操作。 | `self.model.save_pretrained`, `self.tokenizer.save_pretrained`, `logger.info` |
| [`IterativeAlignment.get_history`](../saddlellm/ExclusiveTechniques.py#L645)<br><sub>`get_history(self) -> List[Dict]`</sub> | method | `IterativeAlignment` 中读取`get_history`的公开操作。 | — |
| [`TestTimeCompute.__init__`](../saddlellm/ExclusiveTechniques.py#L691)<br><sub>`__init__(self, model, tokenizer)`</sub> | method | 初始化 `TestTimeCompute` 实例及其运行依赖。 | `next`, `model.parameters` |
| [`TestTimeCompute.solve`](../saddlellm/ExclusiveTechniques.py#L697)<br><sub>`solve(self, problem: str, level: int=3, custom_verify_fn: Callable=None) -> Dict`</sub> | method | 推理时计算缩放求解。 | `self.LEVEL_CONFIGS.get`, `logger.info`, `range`, `self._generate`, `candidates.append`, `self._parse_response`, `self._extract_score`, `verified.append`, `candidates.sort`, `x.get` |
| [`TestTimeCompute._parse_response`](../saddlellm/ExclusiveTechniques.py#L785)<br><sub>`_parse_response(self, text: str) -> Dict`</sub> | method | 从模型输出中分离推理和答案。 | `text.rfind`, `strip` |
| [`TestTimeCompute._extract_score`](../saddlellm/ExclusiveTechniques.py#L800)<br><sub>`_extract_score(self, text: str) -> int`</sub> | method | 从文本中提取数字评分。 | `re.search`, `min`, `max`, `int`, `match.group` |
| [`TestTimeCompute._generate`](../saddlellm/ExclusiveTechniques.py#L812)<br><sub>`_generate(self, prompt: str, temperature: float=0.3) -> str`</sub> | method | `TestTimeCompute` 中生成`generate`的内部辅助逻辑。 | `self.model.eval`, `to`, `self.tokenizer`, `len`, `self.tokenizer.decode`, `self.model.generate`, `strip` |
| [`RLAIF.__init__`](../saddlellm/ExclusiveTechniques.py#L848)<br><sub>`__init__(self, policy_model, judge_model=None, tokenizer=None, constitution: List[str]=None)`</sub> | method | 初始化 `RLAIF` 实例及其运行依赖。 | `next`, `policy_model.parameters` |
| [`RLAIF.judge_pair`](../saddlellm/ExclusiveTechniques.py#L870)<br><sub>`judge_pair(self, prompt: str, response_a: str, response_b: str) -> Dict`</sub> | method | AI 裁判判断两个回复哪个更好。 | `random.sample`, `min`, `len`, `join`, `chr`, `enumerate`, `self._generate`, `self._extract_confidence` |
| [`RLAIF.generate_preference_data`](../saddlellm/ExclusiveTechniques.py#L919)<br><sub>`generate_preference_data(self, prompts: List[str], num_pairs_per_prompt: int=4) -> List[Dict]`</sub> | method | 自动生成偏好对训练数据 — 完全不需要人工。 | `RejectionSampling`, `rs.generate_candidates`, `len`, `any`, `scored.append`, `scored.sort`, `self.judge_pair`, `data.append`, `logger.info` |
| [`RLAIF._generate`](../saddlellm/ExclusiveTechniques.py#L975)<br><sub>`_generate(self, prompt: str, temperature: float=0.3) -> str`</sub> | method | `RLAIF` 中生成`generate`的内部辅助逻辑。 | `self.judge_model.eval`, `to`, `self.tokenizer`, `torch.no_grad`, `self.judge_model.generate`, `self.tokenizer.decode`, `len`, `strip` |
| [`RLAIF._extract_confidence`](../saddlellm/ExclusiveTechniques.py#L990)<br><sub>`_extract_confidence(self, text: str) -> float`</sub> | method | `RLAIF` 中提取`extract_confidence`的内部辅助逻辑。 | `re.search`, `min`, `int`, `match.group` |
| [`SLURPMixer.__init__`](../saddlellm/ExclusiveTechniques.py#L1034)<br><sub>`__init__(self, text_data, instruct_data=None, chat_data=None, code_data=None)`</sub> | method | 初始化 `SLURPMixer` 实例及其运行依赖。 | — |
| [`SLURPMixer.mix`](../saddlellm/ExclusiveTechniques.py#L1046)<br><sub>`mix(self, ratios: Dict[str, float]=None, seed: int=42)`</sub> | method | 按配比混合数据。 | `ratios.get`, `datasets.append`, `probabilities.append`, `self._format_instruct`, `self._format_chat`, `ValueError`, `sum`, `logger.info`, `dict`, `zip` |
| [`SLURPMixer._format_instruct`](../saddlellm/ExclusiveTechniques.py#L1099)<br><sub>`_format_instruct(self, data)`</sub> | method | 把 instruction-output 对格式化为预训练文本。 | `data.map` |
| [`SLURPMixer._format_instruct._format`](../saddlellm/ExclusiveTechniques.py#L1101)<br><sub>`_format(example)`</sub> | nested function | `SLURPMixer` 中格式化`format`的局部回调/辅助逻辑。 | `example.get` |
| [`SLURPMixer._format_chat`](../saddlellm/ExclusiveTechniques.py#L1108)<br><sub>`_format_chat(self, data)`</sub> | method | 把多轮对话格式化为预训练文本。 | `data.map` |
| [`SLURPMixer._format_chat._format`](../saddlellm/ExclusiveTechniques.py#L1110)<br><sub>`_format(example)`</sub> | nested function | `SLURPMixer` 中格式化`format`的局部回调/辅助逻辑。 | `example.get`, `msg.get`, `text_parts.append`, `join` |
| [`RoPEScaling.ntk_aware`](../saddlellm/ExclusiveTechniques.py#L1147)<br><sub>`ntk_aware(config, target_length: int, original_length: Optional[int]=None, alpha: Optional[float]=None)`</sub> | method | NTK-aware RoPE 扩展。 | `logger.info` |
| [`RoPEScaling.yarn`](../saddlellm/ExclusiveTechniques.py#L1173)<br><sub>`yarn(config, target_length: int, original_length: Optional[int]=None, temperature: float=1.0)`</sub> | method | YaRN (Yet another RoPE extensioN) — 目前最推荐的 RoPE 扩展方法。 | `logger.info` |
| [`RoPEScaling.linear`](../saddlellm/ExclusiveTechniques.py#L1205)<br><sub>`linear(config, target_length: int, original_length: Optional[int]=None)`</sub> | method | 线性 RoPE 插值 (最简单, 但效果通常不如 NTK/YaRN)。 | — |
| [`SparseAutoencoder.__init__`](../saddlellm/ExclusiveTechniques.py#L1247)<br><sub>`__init__(self, input_dim: int, hidden_dim: int, l1_coefficient: float=0.001)`</sub> | method | 初始化 `SparseAutoencoder` 实例及其运行依赖。 | `__init__`, `super`, `nn.Linear`, `nn.init.xavier_uniform_` |
| [`SparseAutoencoder.forward`](../saddlellm/ExclusiveTechniques.py#L1257)<br><sub>`forward(self, x: torch.Tensor)`</sub> | method | Args: x: (batch, input_dim) — 模型某层的激活值 | `F.relu`, `self.encoder`, `self.decoder`, `F.mse_loss`, `mean`, `features.abs` |
| [`SparseAutoencoder.encode`](../saddlellm/ExclusiveTechniques.py#L1280)<br><sub>`encode(self, x: torch.Tensor) -> torch.Tensor`</sub> | method | 将激活值编码为稀疏特征。 | `F.relu`, `self.encoder` |
| [`SparseAutoencoder.get_top_features`](../saddlellm/ExclusiveTechniques.py#L1285)<br><sub>`get_top_features(self, x: torch.Tensor, top_k: int=10) -> Dict`</sub> | method | 找到激活最强的特征。 | `self.encode`, `features.dim`, `features.mean`, `torch.topk`, `indices.tolist`, `values.tolist`, `item`, `mean`, `float` |
| [`SparseAutoencoder.train_on_activations`](../saddlellm/ExclusiveTechniques.py#L1297)<br><sub>`train_on_activations(self, activations: torch.Tensor, epochs: int=100, lr: float=0.001, batch_size: int=256)`</sub> | method | 在模型的激活值上训练 SAE。 | `torch.optim.Adam`, `self.parameters`, `range`, `torch.randperm`, `self.forward`, `optimizer.zero_grad`, `loss.backward`, `optimizer.step`, `loss.item`, `logger.info` |
| [`analyze_model_features`](../saddlellm/ExclusiveTechniques.py#L1331)<br><sub>`analyze_model_features(model, tokenizer, text: str, layer_idx: int=-1)`</sub> | function | 用 SAE 分析模型在处理特定文本时激活了哪些概念。 | `model.modules`, `isinstance`, `target_layer.register_forward_hook`, `tokenizer`, `torch.no_grad`, `model`, `handle.remove`, `list`, `item`, `mean` |
| [`analyze_model_features.hook_fn`](../saddlellm/ExclusiveTechniques.py#L1343)<br><sub>`hook_fn(module, input, output)`</sub> | nested function | 模块级实现`hook_fn`的局部回调/辅助逻辑。 | `isinstance`, `activations.append`, `cpu`, `output.detach` |

## `saddlellm/ExperimentTracker.py`

共 12 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ExperimentTracker.__init__`](../saddlellm/ExperimentTracker.py#L40)<br><sub>`__init__(self, config: TrackingConfig)`</sub> | method | 初始化 `ExperimentTracker` 实例及其运行依赖。 | — |
| [`ExperimentTracker.start`](../saddlellm/ExperimentTracker.py#L47)<br><sub>`start(self)`</sub> | method | `ExperimentTracker` 中实现`start`的公开操作。 | `time.time`, `strftime`, `datetime.now`, `os.path.join`, `os.makedirs`, `SummaryWriter`, `wandb.init`, `logger.warning`, `self._LocalLogger`, `open` |
| [`ExperimentTracker.log_metrics`](../saddlellm/ExperimentTracker.py#L81)<br><sub>`log_metrics(self, metrics: Dict[str, float], step: int)`</sub> | method | 记录指标 | `time.time`, `self._metrics_history.append`, `isinstance`, `self._backend.log`, `hasattr`, `metrics.items`, `self._backend.add_scalar` |
| [`ExperimentTracker.log_hyperparams`](../saddlellm/ExperimentTracker.py#L96)<br><sub>`log_hyperparams(self, params: Dict)`</sub> | method | 记录超参数 | `hasattr`, `json.dumps`, `self._backend.add_text`, `self._backend.config.update`, `os.path.join`, `open`, `json.dump` |
| [`ExperimentTracker.log_artifact`](../saddlellm/ExperimentTracker.py#L108)<br><sub>`log_artifact(self, local_path: str, name: Optional[str]=None)`</sub> | method | 记录产物 (模型文件、图表等) | `os.path.join`, `os.path.basename`, `os.makedirs`, `os.path.dirname`, `os.path.isfile`, `shutil.copy2`, `os.path.isdir`, `shutil.copytree` |
| [`ExperimentTracker.log_model_summary`](../saddlellm/ExperimentTracker.py#L118)<br><sub>`log_model_summary(self, model)`</sub> | method | 记录模型参数量和结构 | `sum`, `p.numel`, `model.parameters`, `round`, `self.log_hyperparams` |
| [`ExperimentTracker.log_gpu_stats`](../saddlellm/ExperimentTracker.py#L131)<br><sub>`log_gpu_stats(self)`</sub> | method | 记录 GPU 状态 | `pynvml.nvmlInit`, `pynvml.nvmlDeviceGetCount`, `range`, `pynvml.nvmlDeviceGetHandleByIndex`, `pynvml.nvmlDeviceGetUtilizationRates`, `pynvml.nvmlDeviceGetMemoryInfo`, `self.log_metrics` |
| [`ExperimentTracker.end`](../saddlellm/ExperimentTracker.py#L149)<br><sub>`end(self)`</sub> | method | 结束追踪 | `time.time`, `os.path.join`, `open`, `json.dump`, `hasattr`, `self._backend.close`, `self._backend.finish`, `logger.info` |
| [`ExperimentTracker.get_metrics_df`](../saddlellm/ExperimentTracker.py#L165)<br><sub>`get_metrics_df(self)`</sub> | method | 返回 pandas DataFrame | `pd.DataFrame` |
| [`ExperimentTracker._LocalLogger.__init__`](../saddlellm/ExperimentTracker.py#L175)<br><sub>`__init__(self, run_dir: str)`</sub> | method | 初始化 `_LocalLogger` 实例及其运行依赖。 | `os.path.join`, `open` |
| [`ExperimentTracker._LocalLogger.log`](../saddlellm/ExperimentTracker.py#L180)<br><sub>`log(self, metrics: Dict, step: int)`</sub> | method | `_LocalLogger` 中记录`log`的公开操作。 | `self._file.write`, `json.dumps`, `self._file.flush` |
| [`ExperimentTracker._LocalLogger.close`](../saddlellm/ExperimentTracker.py#L185)<br><sub>`close(self)`</sub> | method | `_LocalLogger` 中关闭`close`的公开操作。 | `self._file.close` |

## `saddlellm/FactoryBackendPlanner.py`

共 17 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TrainingBackendSpec.to_dict`](../saddlellm/FactoryBackendPlanner.py#L25)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `TrainingBackendSpec` 转为可序列化字典。 | `asdict` |
| [`ParallelismPlan.to_dict`](../saddlellm/FactoryBackendPlanner.py#L50)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ParallelismPlan` 转为可序列化字典。 | `asdict` |
| [`FactoryBackendPlanner.list_backends`](../saddlellm/FactoryBackendPlanner.py#L119)<br><sub>`list_backends(cls) -> Dict[str, Dict]`</sub> | method | `FactoryBackendPlanner` 中列出`list_backends`的公开操作。 | `spec.to_dict`, `cls.BACKENDS.items` |
| [`FactoryBackendPlanner.recommend_parallelism`](../saddlellm/FactoryBackendPlanner.py#L123)<br><sub>`recommend_parallelism(cls, model_params: int, num_gpus: int=1, gpu_memory_gb: float=24.0, seq_length: int=2048, global_batch_size: int=64, micro_batch_size: Optional[int]=None, is_moe: bool=False, prefer_backend: str='auto', training_stage: str='pretrain') -> ParallelismPlan`</sub> | method | `FactoryBackendPlanner` 中实现`recommend_parallelism`的公开操作。 | `max`, `int`, `cls._choose_backend`, `cls._choose_tensor_parallel`, `cls._choose_pipeline_parallel`, `cls._choose_context_parallel`, `cls._choose_expert_parallel`, `cls._default_micro_batch`, `math.ceil`, `cls._estimate_memory` |
| [`FactoryBackendPlanner.recommend_stage`](../saddlellm/FactoryBackendPlanner.py#L201)<br><sub>`recommend_stage(cls, stage: str, model_params: int, num_gpus: int, gpu_memory_gb: float, seq_length: int=2048, is_moe: bool=False) -> Dict`</sub> | method | `FactoryBackendPlanner` 中实现训练阶段的公开操作。 | `cls.recommend_parallelism`, `get`, `plan.to_dict`, `cls.to_training_orchestrator_distributed`, `cls.to_megatron_style_args`, `cls.to_colossal_plugin_spec` |
| [`FactoryBackendPlanner.to_training_orchestrator_distributed`](../saddlellm/FactoryBackendPlanner.py#L236)<br><sub>`to_training_orchestrator_distributed(plan: ParallelismPlan) -> Dict`</sub> | method | `FactoryBackendPlanner` 中实现训练、分布式运行时的公开操作。 | `NotImplementedError`, `strategy_map.get` |
| [`FactoryBackendPlanner.to_megatron_style_args`](../saddlellm/FactoryBackendPlanner.py#L258)<br><sub>`to_megatron_style_args(plan: ParallelismPlan) -> Dict`</sub> | method | `FactoryBackendPlanner` 中实现`to_megatron_style_args`的公开操作。 | — |
| [`FactoryBackendPlanner.to_colossal_plugin_spec`](../saddlellm/FactoryBackendPlanner.py#L271)<br><sub>`to_colossal_plugin_spec(plan: ParallelismPlan) -> Dict`</sub> | method | `FactoryBackendPlanner` 中实现`to_colossal_plugin_spec`的公开操作。 | — |
| [`FactoryBackendPlanner.save_plan`](../saddlellm/FactoryBackendPlanner.py#L298)<br><sub>`save_plan(cls, path: str, plan: ParallelismPlan) -> str`</sub> | method | `FactoryBackendPlanner` 中保存计划的公开操作。 | `os.makedirs`, `os.path.dirname`, `plan.to_dict`, `cls.to_training_orchestrator_distributed`, `cls.to_megatron_style_args`, `cls.to_colossal_plugin_spec`, `open`, `json.dump` |
| [`FactoryBackendPlanner._choose_backend`](../saddlellm/FactoryBackendPlanner.py#L311)<br><sub>`_choose_backend(model_params: int, total_gpus: int, gpu_memory_gb: float, is_moe: bool, prefer_backend: str, training_stage: str) -> str`</sub> | method | `FactoryBackendPlanner` 中实现后端的内部辅助逻辑。 | `ValueError` |
| [`FactoryBackendPlanner._choose_tensor_parallel`](../saddlellm/FactoryBackendPlanner.py#L339)<br><sub>`_choose_tensor_parallel(model_params: int, total_gpus: int, backend: str) -> int`</sub> | method | `FactoryBackendPlanner` 中实现`choose_tensor_parallel`的内部辅助逻辑。 | `FactoryBackendPlanner._largest_factor` |
| [`FactoryBackendPlanner._choose_pipeline_parallel`](../saddlellm/FactoryBackendPlanner.py#L347)<br><sub>`_choose_pipeline_parallel(model_params: int, remaining_gpus: int, backend: str) -> int`</sub> | method | `FactoryBackendPlanner` 中实现`choose_pipeline_parallel`的内部辅助逻辑。 | `FactoryBackendPlanner._largest_factor` |
| [`FactoryBackendPlanner._choose_context_parallel`](../saddlellm/FactoryBackendPlanner.py#L355)<br><sub>`_choose_context_parallel(seq_length: int, remaining_gpus: int, backend: str) -> int`</sub> | method | `FactoryBackendPlanner` 中实现`choose_context_parallel`的内部辅助逻辑。 | — |
| [`FactoryBackendPlanner._choose_expert_parallel`](../saddlellm/FactoryBackendPlanner.py#L364)<br><sub>`_choose_expert_parallel(is_moe: bool, remaining_gpus: int, backend: str) -> int`</sub> | method | `FactoryBackendPlanner` 中实现`choose_expert_parallel`的内部辅助逻辑。 | `FactoryBackendPlanner._largest_factor`, `min` |
| [`FactoryBackendPlanner._default_micro_batch`](../saddlellm/FactoryBackendPlanner.py#L370)<br><sub>`_default_micro_batch(model_params: int, seq_length: int, gpu_memory_gb: float) -> int`</sub> | method | `FactoryBackendPlanner` 中实现批次的内部辅助逻辑。 | — |
| [`FactoryBackendPlanner._estimate_memory`](../saddlellm/FactoryBackendPlanner.py#L379)<br><sub>`_estimate_memory(model_params: int, seq_length: int, micro_batch_size: int, data_parallel: int, tensor_parallel: int, pipeline_parallel: int, expert_parallel: int, context_parallel: int, backend: str) -> (float, float)`</sub> | method | `FactoryBackendPlanner` 中估算`estimate_memory`的内部辅助逻辑。 | `max` |
| [`FactoryBackendPlanner._largest_factor`](../saddlellm/FactoryBackendPlanner.py#L406)<br><sub>`_largest_factor(value: int, max_factor: int) -> int`</sub> | method | `FactoryBackendPlanner` 中实现`largest_factor`的内部辅助逻辑。 | `max`, `int`, `min`, `range` |

## `saddlellm/FrontierAlign.py`

共 25 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ConstitutionalAI.__init__`](../saddlellm/FrontierAlign.py#L91)<br><sub>`__init__(self, model, tokenizer, config: CAIConfig=None)`</sub> | method | 初始化 `ConstitutionalAI` 实例及其运行依赖。 | `CAIConfig`, `next`, `model.parameters` |
| [`ConstitutionalAI.generate_with_cai`](../saddlellm/FrontierAlign.py#L97)<br><sub>`generate_with_cai(self, prompt: str, initial_response: str=None) -> Dict`</sub> | method | 用 CAI 流程生成高质量回复。 | `self._generate`, `range`, `self._generate_critiques`, `self._extract_violations`, `extend`, `logger.info`, `self._revise_response`, `append`, `self._similarity`, `len` |
| [`ConstitutionalAI._generate_critiques`](../saddlellm/FrontierAlign.py#L157)<br><sub>`_generate_critiques(self, prompt: str, response: str) -> str`</sub> | method | 根据宪法规则批评回复。 | `random.sample`, `min`, `len`, `join`, `chr`, `enumerate`, `self._generate` |
| [`ConstitutionalAI._revise_response`](../saddlellm/FrontierAlign.py#L180)<br><sub>`_revise_response(self, prompt: str, response: str, critiques: str) -> str`</sub> | method | 基于批评生成修订版回复。 | `self._generate` |
| [`ConstitutionalAI._extract_violations`](../saddlellm/FrontierAlign.py#L202)<br><sub>`_extract_violations(self, critique_text: str) -> List[str]`</sub> | method | 从批评文本中提取违规项。 | `critique_text.split`, `line.strip`, `any`, `violations.append` |
| [`ConstitutionalAI._similarity`](../saddlellm/FrontierAlign.py#L216)<br><sub>`_similarity(self, a: str, b: str) -> float`</sub> | method | 简单文本相似度 (Jaccard on words)。 | `set`, `a.split`, `b.split`, `len` |
| [`ConstitutionalAI._generate`](../saddlellm/FrontierAlign.py#L224)<br><sub>`_generate(self, prompt: str, temperature: float=0.7) -> str`</sub> | method | `ConstitutionalAI` 中生成`generate`的内部辅助逻辑。 | `self.model.eval`, `to`, `self.tokenizer`, `torch.no_grad`, `self.model.generate`, `self.tokenizer.decode`, `len`, `strip` |
| [`ConstitutionalAI.generate_sft_data`](../saddlellm/FrontierAlign.py#L246)<br><sub>`generate_sft_data(self, prompts: List[str], samples_per_prompt: int=2) -> List[Dict]`</sub> | method | 用 CAI 自动生成高质量 SFT 训练数据。 | `enumerate`, `logger.info`, `len`, `range`, `self.generate_with_cai`, `data.append` |
| [`ProcessRewardModel.__init__`](../saddlellm/FrontierAlign.py#L329)<br><sub>`__init__(self, model=None, tokenizer=None, config: PRMConfig=None)`</sub> | method | 初始化 `ProcessRewardModel` 实例及其运行依赖。 | `PRMConfig`, `next`, `model.parameters` |
| [`ProcessRewardModel.split_into_steps`](../saddlellm/FrontierAlign.py#L335)<br><sub>`split_into_steps(self, text: str) -> List[str]`</sub> | method | 将文本自动拆分为推理步骤。 | `remaining.split`, `p.strip`, `len`, `re.split`, `s.strip`, `max`, `range` |
| [`ProcessRewardModel.score_steps_by_rules`](../saddlellm/FrontierAlign.py#L365)<br><sub>`score_steps_by_rules(self, question: str, steps: List[str]) -> List[Dict]`</sub> | method | 基于规则对推理步骤打分 (不需要额外模型)。 | `enumerate`, `re.search`, `any`, `self._has_overlap`, `prev_step.split`, `len`, `self._similarity`, `max`, `min`, `scores.append` |
| [`ProcessRewardModel.score_steps_by_self_check`](../saddlellm/FrontierAlign.py#L416)<br><sub>`score_steps_by_self_check(self, question: str, steps: List[str]) -> List[Dict]`</sub> | method | 让模型自己检查每一步推理 (需要模型)。 | `logger.warning`, `self.score_steps_by_rules`, `self.model.eval`, `enumerate`, `join`, `chr`, `len`, `to`, `self.tokenizer`, `torch.no_grad` |
| [`ProcessRewardModel.aggregate_scores`](../saddlellm/FrontierAlign.py#L480)<br><sub>`aggregate_scores(self, step_scores: List[Dict], method: str='min') -> float`</sub> | method | 聚合各步骤分数为整体分数。 | `min`, `sum`, `len` |
| [`ProcessRewardModel.filter_best_reasoning`](../saddlellm/FrontierAlign.py#L500)<br><sub>`filter_best_reasoning(self, question: str, candidate_reasonings: List[List[str]], threshold: float=0.7) -> List[Dict]`</sub> | method | 从多个推理候选中选择过程最正确的。 | `enumerate`, `self.score_steps_by_rules`, `self.aggregate_scores`, `results.append`, `results.sort` |
| [`ProcessRewardModel._has_overlap`](../saddlellm/FrontierAlign.py#L527)<br><sub>`_has_overlap(self, text: str, tokens: List[str]) -> bool`</sub> | method | 检查 text 是否包含 tokens 中的词。 | `text.lower`, `any`, `t.lower` |
| [`ProcessRewardModel._similarity`](../saddlellm/FrontierAlign.py#L532)<br><sub>`_similarity(self, a: str, b: str) -> float`</sub> | method | `ProcessRewardModel` 中实现`similarity`的内部辅助逻辑。 | `set`, `a.split`, `b.split`, `len` |
| [`RejectionSampling.__init__`](../saddlellm/FrontierAlign.py#L577)<br><sub>`__init__(self, model, tokenizer, config: RejectionSamplingConfig=None)`</sub> | method | 初始化 `RejectionSampling` 实例及其运行依赖。 | `RejectionSamplingConfig`, `next`, `model.parameters` |
| [`RejectionSampling.generate_candidates`](../saddlellm/FrontierAlign.py#L583)<br><sub>`generate_candidates(self, prompt: str) -> List[str]`</sub> | method | 为一个 prompt 生成多个候选回复。 | `self.model.eval`, `range`, `to`, `self.tokenizer`, `torch.no_grad`, `self.model.generate`, `self.tokenizer.decode`, `len`, `candidates.append`, `strip` |
| [`RejectionSampling.select_best`](../saddlellm/FrontierAlign.py#L607)<br><sub>`select_best(self, candidates: List[str], prompt: str='', prm: Optional['ProcessRewardModel']=None, constitution: Optional[List[str]]=None, reward_fn: Optional[Callable]=None, method: str='combined') -> Dict`</sub> | method | 从候选中选择最佳回复。 | `enumerate`, `prm.score_steps_by_rules`, `prm.split_into_steps`, `max`, `self._constitution_score`, `self._rule_based_score`, `reward_fn`, `sum`, `scores.values`, `len` |
| [`RejectionSampling._constitution_score`](../saddlellm/FrontierAlign.py#L688)<br><sub>`_constitution_score(self, text: str, constitution: List[str]) -> float`</sub> | method | 基于宪法规则打分: 违反越少分越高。 | `self._extract_keywords`, `text.lower`, `kw.lower`, `min` |
| [`RejectionSampling._rule_based_score`](../saddlellm/FrontierAlign.py#L700)<br><sub>`_rule_based_score(self, text: str) -> float`</sub> | method | 启发式质量评分。 | `len`, `any`, `text.lower`, `max`, `min` |
| [`RejectionSampling._extract_keywords`](../saddlellm/FrontierAlign.py#L729)<br><sub>`_extract_keywords(self, rule: str) -> List[str]`</sub> | method | 从规则中提取关键词。 | `split`, `replace`, `rule.replace`, `len` |
| [`FrontierAlignmentPipeline.__init__`](../saddlellm/FrontierAlign.py#L762)<br><sub>`__init__(self, model, tokenizer, configs: Dict=None)`</sub> | method | 初始化 `FrontierAlignmentPipeline` 实例及其运行依赖。 | `RejectionSampling`, `cfg.get`, `ProcessRewardModel`, `ConstitutionalAI`, `next`, `model.parameters` |
| [`FrontierAlignmentPipeline.query`](../saddlellm/FrontierAlign.py#L772)<br><sub>`query(self, prompt: str, num_candidates: int=8) -> Dict`</sub> | method | 通过完整流水线生成高质量回复。 | `self.rs.generate_candidates`, `enumerate`, `self.prm.split_into_steps`, `self.prm.score_steps_by_rules`, `self.prm.aggregate_scores`, `prm_results.append`, `len`, `prm_results.sort`, `min`, `self.cai.generate_with_cai` |
| [`FrontierAlignmentPipeline.generate_training_data`](../saddlellm/FrontierAlign.py#L822)<br><sub>`generate_training_data(self, prompts: List[str], samples_per_prompt: int=2, quality_threshold: float=0.6) -> List[Dict]`</sub> | method | 批量生成高质量 SFT 训练数据。 | `enumerate`, `logger.info`, `len`, `range`, `self.query`, `result.get`, `training_data.append` |

## `saddlellm/GRPOTrainer.py`

共 26 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`GRPOConfig.validate`](../saddlellm/GRPOTrainer.py#L92)<br><sub>`validate(self) -> None`</sub> | method | `GRPOConfig` 中校验`validate`的公开操作。 | `ValueError` |
| [`GRPOTrainer.__init__`](../saddlellm/GRPOTrainer.py#L145)<br><sub>`__init__(self, model, ref_model, tokenizer, config: Optional[GRPOConfig]=None)`</sub> | method | 初始化 `GRPOTrainer` 实例及其运行依赖。 | `GRPOConfig`, `self.config.validate`, `callable`, `getattr`, `ValueError`, `self.ref_model.parameters`, `self.ref_model.eval`, `self.model.parameters`, `torch.optim.AdamW`, `next` |
| [`GRPOTrainer.train`](../saddlellm/GRPOTrainer.py#L196)<br><sub>`train(self, prompts: List[str], reward_fn: RewardFn, num_steps: int=1000, reward_model: Optional[RewardFn]=None, eval_prompts: Optional[List[str]]=None) -> List[Dict[str, float]]`</sub> | method | Train on text prompts using a deterministic/verifiable reward. | `ValueError`, `logger.info`, `self.optimizer.zero_grad`, `self.model.train`, `range`, `min`, `len`, `self._sample_prompts`, `self._generate_responses_with_log_probs`, `zip` |
| [`GRPOTrainer._correct_partial_accumulation`](../saddlellm/GRPOTrainer.py#L397)<br><sub>`_correct_partial_accumulation(self, accumulated_steps: int) -> None`</sub> | method | Turn a final partial sum of ``loss / target`` into its actual mean. | `torch.no_grad`, `self.model.parameters`, `parameter.grad.mul_` |
| [`GRPOTrainer._generate_responses`](../saddlellm/GRPOTrainer.py#L409)<br><sub>`_generate_responses(self, prompts: Sequence[str]) -> List[str]`</sub> | method | `GRPOTrainer` 中生成`generate_responses`的内部辅助逻辑。 | `self._generate_responses_with_log_probs` |
| [`GRPOTrainer._generate_responses_with_log_probs`](../saddlellm/GRPOTrainer.py#L413)<br><sub>`_generate_responses_with_log_probs(self, prompts: Sequence[str])`</sub> | method | Generate completions; return texts, token ids, and per-token log probabilities of the sampled tokens under the rollout policy. | `self.model.eval`, `getattr`, `to`, `self.tokenizer`, `list`, `generation_kwargs.update`, `torch.inference_mode`, `self.model.generate`, `self.model`, `torch.cat` |
| [`GRPOTrainer._compute_group_advantages`](../saddlellm/GRPOTrainer.py#L512)<br><sub>`_compute_group_advantages(self, rewards: Sequence[float]) -> List[float]`</sub> | method | `GRPOTrainer` 中计算`compute_group_advantages`的内部辅助逻辑。 | `torch.tensor`, `torch.clamp`, `rewards_t.mean`, `rewards_t.std`, `centered.tolist` |
| [`GRPOTrainer._tokenize_trajectory`](../saddlellm/GRPOTrainer.py#L527)<br><sub>`_tokenize_trajectory(self, prompt: str, response: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]`</sub> | method | `GRPOTrainer` 中分词`tokenize_trajectory`的内部辅助逻辑。 | `self.tokenizer`, `len`, `torch.tensor`, `torch.ones_like` |
| [`GRPOTrainer._completion_log_probs`](../saddlellm/GRPOTrainer.py#L554)<br><sub>`_completion_log_probs(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor`</sub> | method | `GRPOTrainer` 中记录`completion_log_probs`的内部辅助逻辑。 | `F.log_softmax`, `completion_logits.float`, `squeeze`, `torch.gather`, `completion_labels.unsqueeze` |
| [`GRPOTrainer._reference_logits`](../saddlellm/GRPOTrainer.py#L566)<br><sub>`_reference_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Optional[torch.Tensor]`</sub> | method | `GRPOTrainer` 中实现`reference_logits`的内部辅助逻辑。 | `torch.no_grad`, `self.ref_model`, `self.model.disable_adapter`, `self.model` |
| [`GRPOTrainer._compute_grpo_loss`](../saddlellm/GRPOTrainer.py#L585)<br><sub>`_compute_grpo_loss(self, prompt: str, response: str, advantage: float, response_ids: Optional[torch.Tensor]=None, old_log_probs: Optional[torch.Tensor]=None, padded_prompt_ids: Optional[torch.Tensor]=None, padded_prompt_mask: Optional[torch.Tensor]=None, prompt_width: Optional[int]=None) -> Optional[Tuple[torch.Tensor, float]]`</sub> | method | `GRPOTrainer` 中计算`compute_grpo_loss`的内部辅助逻辑。 | `to`, `response_ids.numel`, `unsqueeze`, `torch.cat`, `torch.ones_like`, `self._tokenize_trajectory`, `self.model`, `self._completion_log_probs`, `policy_log_probs.detach`, `torch.clamp` |
| [`GRPOTrainer._estimate_kl`](../saddlellm/GRPOTrainer.py#L661)<br><sub>`_estimate_kl(self, ref_policy_log_ratio: torch.Tensor) -> torch.Tensor`</sub> | method | `GRPOTrainer` 中估算`estimate_kl`的内部辅助逻辑。 | `ref_policy_log_ratio.square`, `torch.exp` |
| [`GRPOTrainer._sample_prompts`](../saddlellm/GRPOTrainer.py#L668)<br><sub>`_sample_prompts(self, prompts: Sequence[str], batch_size: int) -> List[str]`</sub> | method | `GRPOTrainer` 中采样`sample_prompts`的内部辅助逻辑。 | `min`, `len`, `self._rng.sample`, `range`, `tuple`, `list`, `self._rng.shuffle`, `indices.extend` |
| [`GRPOTrainer._update_ref_model`](../saddlellm/GRPOTrainer.py#L690)<br><sub>`_update_ref_model(self) -> None`</sub> | method | `GRPOTrainer` 中更新模型的内部辅助逻辑。 | `logger.debug`, `self.ref_model.load_state_dict`, `self.model.state_dict`, `self.ref_model.eval` |
| [`GRPOTrainer.evaluate`](../saddlellm/GRPOTrainer.py#L699)<br><sub>`evaluate(self, eval_prompts: List[str], reward_fn: RewardFn) -> Dict[str, float]`</sub> | method | `GRPOTrainer` 中评估`evaluate`的公开操作。 | `self._generate_responses`, `float`, `reward_fn`, `best_rewards.append`, `max`, `all_rewards.extend`, `sum`, `len` |
| [`GRPOTrainer.profile_prompts`](../saddlellm/GRPOTrainer.py#L719)<br><sub>`profile_prompts(self, prompts: Sequence[str], reward_fn: RewardFn, max_prompts: Optional[int]=None) -> List[Dict[str, object]]`</sub> | method | Measure source-side group learnability before training. | `list`, `self._generate_responses`, `float`, `reward_fn`, `torch.tensor`, `reward_tensor.std`, `profiles.append`, `reward_tensor.mean` |
| [`GRPOTrainer.save`](../saddlellm/GRPOTrainer.py#L756)<br><sub>`save(self, path: str) -> None`</sub> | method | `GRPOTrainer` 中保存`save`的公开操作。 | `Path`, `output.mkdir`, `self.model.save_pretrained`, `self.tokenizer.save_pretrained`, `logger.info` |
| [`GRPOTrainer.get_metrics_df`](../saddlellm/GRPOTrainer.py#L763)<br><sub>`get_metrics_df(self)`</sub> | method | `GRPOTrainer` 中读取指标的公开操作。 | `pd.DataFrame` |
| [`GRPOTrainer.reward_length`](../saddlellm/GRPOTrainer.py#L772)<br><sub>`reward_length(target_min: int=100, target_max: int=2000, penalty_rate: float=0.001) -> RewardFn`</sub> | method | `GRPOTrainer` 中实现`reward_length`的公开操作。 | — |
| [`GRPOTrainer.reward_length._fn`](../saddlellm/GRPOTrainer.py#L777)<br><sub>`_fn(prompt: str, response: str) -> float`</sub> | nested function | `GRPOTrainer` 中实现`fn`的局部回调/辅助逻辑。 | `len`, `max` |
| [`GRPOTrainer.reward_format`](../saddlellm/GRPOTrainer.py#L789)<br><sub>`reward_format(required_patterns: Optional[List[str]]=None, forbidden_patterns: Optional[List[str]]=None) -> RewardFn`</sub> | method | `GRPOTrainer` 中格式化`reward_format`的公开操作。 | — |
| [`GRPOTrainer.reward_format._fn`](../saddlellm/GRPOTrainer.py#L796)<br><sub>`_fn(prompt: str, response: str) -> float`</sub> | nested function | `GRPOTrainer` 中实现`fn`的局部回调/辅助逻辑。 | `re.search`, `max` |
| [`GRPOTrainer.reward_combined`](../saddlellm/GRPOTrainer.py#L810)<br><sub>`reward_combined(*reward_fns: RewardFn, weights: Optional[List[float]]=None) -> RewardFn`</sub> | method | `GRPOTrainer` 中实现`reward_combined`的公开操作。 | `len`, `ValueError` |
| [`GRPOTrainer.reward_combined._fn`](../saddlellm/GRPOTrainer.py#L818)<br><sub>`_fn(prompt: str, response: str) -> float`</sub> | nested function | `GRPOTrainer` 中实现`fn`的局部回调/辅助逻辑。 | `sum`, `reward_fn`, `zip` |
| [`GRPOTrainer.reward_math_accuracy`](../saddlellm/GRPOTrainer.py#L827)<br><sub>`reward_math_accuracy(extract_answer: Optional[Callable[[str], str]]=None) -> RewardFn`</sub> | method | Build a simple exact-answer reward for math-style prompts. | `re.compile` |
| [`GRPOTrainer.reward_math_accuracy._fn`](../saddlellm/GRPOTrainer.py#L837)<br><sub>`_fn(prompt: str, response: str) -> float`</sub> | nested function | `GRPOTrainer` 中实现`fn`的局部回调/辅助逻辑。 | `extract_answer`, `float`, `strip`, `str`, `answer_pattern.search`, `expected_match.group`, `predicted_match.group` |

## `saddlellm/InferenceServer.py`

共 18 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`InferenceServerSettings.__post_init__`](../saddlellm/InferenceServer.py#L36)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `InferenceServerSettings` 创建后校验并规范化字段。 | `ValueError` |
| [`LocalGenerationEngine.__init__`](../saddlellm/InferenceServer.py#L79)<br><sub>`__init__(self, model: Any, tokenizer: Any, settings: InferenceServerSettings)`</sub> | method | 初始化 `LocalGenerationEngine` 实例及其运行依赖。 | `self._infer_model_name` |
| [`LocalGenerationEngine.chat`](../saddlellm/InferenceServer.py#L85)<br><sub>`chat(self, request: ChatCompletionRequest) -> Dict[str, Any]`</sub> | method | `LocalGenerationEngine` 中实现`chat`的公开操作。 | `message.model_dump`, `ValueError`, `self._chat_prompt`, `self._generate`, `uuid.uuid4`, `int`, `time.time` |
| [`LocalGenerationEngine.complete`](../saddlellm/InferenceServer.py#L117)<br><sub>`complete(self, request: CompletionRequest) -> Dict[str, Any]`</sub> | method | `LocalGenerationEngine` 中实现`complete`的公开操作。 | `ValueError`, `self._generate`, `uuid.uuid4`, `int`, `time.time` |
| [`LocalGenerationEngine._chat_prompt`](../saddlellm/InferenceServer.py#L143)<br><sub>`_chat_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str`</sub> | method | `LocalGenerationEngine` 中实现提示词的内部辅助逻辑。 | `getattr`, `callable`, `apply_template`, `list`, `parts.append`, `join` |
| [`LocalGenerationEngine._generate`](../saddlellm/InferenceServer.py#L160)<br><sub>`_generate(self, prompt: str, *, max_tokens: Optional[int], temperature: float, top_p: float, repetition_penalty: float, stop: Optional[Union[str, List[str]]]) -> Tuple[str, int, int, str]`</sub> | method | `LocalGenerationEngine` 中生成`generate`的内部辅助逻辑。 | `RuntimeError`, `min`, `getattr`, `isinstance`, `max`, `self.tokenizer`, `int`, `self._input_device`, `hasattr`, `value.to` |
| [`LocalGenerationEngine._input_device`](../saddlellm/InferenceServer.py#L220)<br><sub>`_input_device(self)`</sub> | method | `LocalGenerationEngine` 中实现输入、设备的内部辅助逻辑。 | `getattr`, `str`, `next`, `self.model.parameters` |
| [`LocalGenerationEngine._apply_stop`](../saddlellm/InferenceServer.py#L230)<br><sub>`_apply_stop(text: str, stop: Optional[Union[str, List[str]]]) -> Tuple[str, bool]`</sub> | method | `LocalGenerationEngine` 中应用`apply_stop`的内部辅助逻辑。 | `isinstance`, `text.find`, `min` |
| [`LocalGenerationEngine._infer_model_name`](../saddlellm/InferenceServer.py#L243)<br><sub>`_infer_model_name(model_path: str) -> str`</sub> | method | `LocalGenerationEngine` 中推断模型的内部辅助逻辑。 | `model_path.rstrip`, `os.path.basename` |
| [`create_inference_app`](../saddlellm/InferenceServer.py#L250)<br><sub>`create_inference_app(settings: Optional[InferenceServerSettings]=None, *, model: Any=None, tokenizer: Any=None)`</sub> | function | Create a FastAPI app, optionally with an already loaded model for tests. | `ImportError`, `InferenceServerSettings`, `ValueError`, `load_inference_model`, `LocalGenerationEngine`, `asyncio.Semaphore`, `max`, `_load_release_metadata`, `FastAPI` |
| [`create_inference_app.authorize`](../saddlellm/InferenceServer.py#L277)<br><sub>`async authorize(authorization: Optional[str]=Header(default=None)) -> None`</sub> | nested function | 模块级鉴权`authorize`的局部回调/辅助逻辑。 | `partition`, `scheme.lower`, `secrets.compare_digest`, `HTTPException` |
| [`create_inference_app.health`](../saddlellm/InferenceServer.py#L285)<br><sub>`async health() -> Dict[str, Any]`</sub> | nested function | 模块级实现`health`的局部回调/辅助逻辑。 | `release_metadata.get`, `isinstance`, `gate.get` |
| [`create_inference_app.models`](../saddlellm/InferenceServer.py#L295)<br><sub>`async models() -> Dict[str, Any]`</sub> | nested function | 模块级实现`models`的局部回调/辅助逻辑。 | — |
| [`create_inference_app.chat_completions`](../saddlellm/InferenceServer.py#L309)<br><sub>`async chat_completions(request: ChatCompletionRequest) -> Dict[str, Any]`</sub> | nested function | 模块级实现`chat_completions`的局部回调/辅助逻辑。 | `HTTPException`, `asyncio.to_thread`, `str` |
| [`create_inference_app.completions`](../saddlellm/InferenceServer.py#L319)<br><sub>`async completions(request: CompletionRequest) -> Dict[str, Any]`</sub> | nested function | 模块级实现`completions`的局部回调/辅助逻辑。 | `HTTPException`, `asyncio.to_thread`, `str` |
| [`load_inference_model`](../saddlellm/InferenceServer.py#L331)<br><sub>`load_inference_model(settings: InferenceServerSettings) -> Tuple[Any, Any]`</sub> | function | Load a full checkpoint or PEFT adapter for local generation. | `ValueError`, `ImportError`, `torch.cuda.is_available`, `RuntimeError`, `expanduser`, `Path`, `adapter_config_path.is_file`, `stabilize_peft_optional_backends`, `logger.warning`, `json.loads` |
| [`_has_tokenizer_files`](../saddlellm/InferenceServer.py#L415)<br><sub>`_has_tokenizer_files(path: Path) -> bool`</sub> | function | 模块级实现分词器的内部辅助逻辑。 | `any`, `is_file` |
| [`_load_release_metadata`](../saddlellm/InferenceServer.py#L428)<br><sub>`_load_release_metadata(model_path: str) -> Dict[str, Any]`</sub> | function | 模块级加载发布包的内部辅助逻辑。 | `expanduser`, `Path`, `manifest.is_file`, `json.loads`, `manifest.read_text`, `isinstance` |

## `saddlellm/LLMLoader.py`

共 29 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`LLMLoader.__init__`](../saddlellm/LLMLoader.py#L46)<br><sub>`__init__(self, config: Union[Dict[str, Any], LLMConfig])`</sub> | method | 初始化加载器 | `isinstance`, `LLMConfig`, `self._init_device` |
| [`LLMLoader._init_device`](../saddlellm/LLMLoader.py#L66)<br><sub>`_init_device(self)`</sub> | method | 初始化设备设置 | `self._auto_select_device`, `self._init_ascend_environment` |
| [`LLMLoader.architecture_support`](../saddlellm/LLMLoader.py#L74)<br><sub>`architecture_support(self) -> Dict[str, Any]`</sub> | method | 检测当前模型路径对应的架构支持级别。 | `to_dict`, `ArchitectureRegistry.detect_from_hf_config` |
| [`LLMLoader._auto_select_device`](../saddlellm/LLMLoader.py#L83)<br><sub>`_auto_select_device(self) -> str`</sub> | method | 自动选择最佳设备 | `check_func` |
| [`LLMLoader._check_cuda_available`](../saddlellm/LLMLoader.py#L99)<br><sub>`_check_cuda_available(self) -> bool`</sub> | method | 检查CUDA是否可用 | `torch.cuda.is_available` |
| [`LLMLoader._check_ascend_available`](../saddlellm/LLMLoader.py#L103)<br><sub>`_check_ascend_available(self) -> bool`</sub> | method | 检查升腾设备是否可用 | `acl.rt.get_device_count` |
| [`LLMLoader._check_mps_available`](../saddlellm/LLMLoader.py#L111)<br><sub>`_check_mps_available(self) -> bool`</sub> | method | 检查Apple MPS是否可用 | `hasattr`, `torch.backends.mps.is_available` |
| [`LLMLoader._check_xpu_available`](../saddlellm/LLMLoader.py#L115)<br><sub>`_check_xpu_available(self) -> bool`</sub> | method | 检查Intel XPU是否可用 | `hasattr`, `torch.xpu.is_available` |
| [`LLMLoader._check_rocm_available`](../saddlellm/LLMLoader.py#L119)<br><sub>`_check_rocm_available(self) -> bool`</sub> | method | 检查AMD ROCm是否可用 | — |
| [`LLMLoader._init_ascend_environment`](../saddlellm/LLMLoader.py#L123)<br><sub>`_init_ascend_environment(self)`</sub> | method | 初始化升腾环境 | `str`, `os.environ.get`, `warnings.warn`, `RuntimeError` |
| [`LLMLoader._get_compute_dtype`](../saddlellm/LLMLoader.py#L140)<br><sub>`_get_compute_dtype(self) -> torch.dtype`</sub> | method | 获取计算数据类型 | `getattr`, `precision_map.get`, `torch.cuda.is_available` |
| [`LLMLoader._get_quantization_config`](../saddlellm/LLMLoader.py#L167)<br><sub>`_get_quantization_config(self) -> Optional[BitsAndBytesConfig]`</sub> | method | 获取量化配置 | `warnings.warn`, `BitsAndBytesConfig`, `self._get_compute_dtype` |
| [`LLMLoader._get_device_map`](../saddlellm/LLMLoader.py#L183)<br><sub>`_get_device_map(self) -> Optional[Dict[str, Any]]`</sub> | method | 获取设备映射(用于多GPU) | `self._parse_gpu_ids`, `PretrainedConfig.from_pretrained`, `getattr`, `len`, `enumerate`, `device_map.update`, `range` |
| [`LLMLoader._parse_gpu_ids`](../saddlellm/LLMLoader.py#L223)<br><sub>`_parse_gpu_ids(self) -> List[int]`</sub> | method | 解析GPU ID字符串 | `int`, `id_str.strip`, `self.config.gpu_ids.split`, `warnings.warn` |
| [`LLMLoader._get_max_memory`](../saddlellm/LLMLoader.py#L234)<br><sub>`_get_max_memory(self) -> Optional[Dict[int, str]]`</sub> | method | 获取最大内存配置 | `self._parse_gpu_ids` |
| [`LLMLoader._standard_device_map_arg`](../saddlellm/LLMLoader.py#L245)<br><sub>`_standard_device_map_arg(self, device_map)`</sub> | method | `LLMLoader` 中实现设备的内部辅助逻辑。 | — |
| [`LLMLoader._pipeline_device_arg`](../saddlellm/LLMLoader.py#L252)<br><sub>`_pipeline_device_arg(self, device_map)`</sub> | method | `LLMLoader` 中实现设备的内部辅助逻辑。 | — |
| [`LLMLoader._supports_flash_attention`](../saddlellm/LLMLoader.py#L261)<br><sub>`_supports_flash_attention(self) -> bool`</sub> | method | `LLMLoader` 中实现注意力的内部辅助逻辑。 | — |
| [`LLMLoader._load_ascend_model`](../saddlellm/LLMLoader.py#L270)<br><sub>`_load_ascend_model(self)`</sub> | method | 加载升腾模型 | `InferSession`, `AutoTokenizer.from_pretrained`, `RuntimeError` |
| [`LLMLoader._load_standard_model`](../saddlellm/LLMLoader.py#L290)<br><sub>`_load_standard_model(self)`</sub> | method | 加载标准模型(HuggingFace格式) | `self._get_quantization_config`, `self._get_device_map`, `self._get_max_memory`, `self._standard_device_map_arg`, `self._get_compute_dtype`, `self._supports_flash_attention`, `AutoModelForCausalLM.from_pretrained`, `self.model.to`, `AutoTokenizer.from_pretrained`, `pipeline` |
| [`LLMLoader.load_model`](../saddlellm/LLMLoader.py#L335)<br><sub>`load_model(self)`</sub> | method | 加载模型和tokenizer | `self._load_ascend_model`, `self._load_standard_model` |
| [`LLMLoader._generate_with_ascend`](../saddlellm/LLMLoader.py#L343)<br><sub>`_generate_with_ascend(self, prompt: str, **kwargs) -> str`</sub> | method | 使用升腾生成文本 | `self.tokenizer.encode`, `kwargs.get`, `len`, `self.model.infer`, `self.tokenizer.decode`, `RuntimeError` |
| [`LLMLoader._generate_standard`](../saddlellm/LLMLoader.py#L365)<br><sub>`_generate_standard(self, prompt: str, **kwargs) -> str`</sub> | method | 使用标准模型生成文本 | `RuntimeError`, `next`, `self.model.parameters`, `to`, `self.tokenizer`, `kwargs.get`, `torch.no_grad`, `self.model.generate`, `self.tokenizer.decode`, `generated.startswith` |
| [`LLMLoader.generate`](../saddlellm/LLMLoader.py#L389)<br><sub>`generate(self, prompt: str, **kwargs) -> str`</sub> | method | 生成文本 | `RuntimeError`, `self._generate_with_ascend`, `self._generate_standard` |
| [`LLMLoader.chat`](../saddlellm/LLMLoader.py#L411)<br><sub>`chat(self, messages: List[Dict[str, str]], **kwargs) -> str`</sub> | method | 使用 chat_template 进行多轮消息生成。 | `RuntimeError`, `hasattr`, `self.tokenizer.apply_chat_template`, `join`, `m.get`, `self.generate` |
| [`LLMLoader.unload`](../saddlellm/LLMLoader.py#L426)<br><sub>`unload(self)`</sub> | method | 释放模型资源。 | `gc.collect`, `torch.cuda.is_available`, `torch.cuda.empty_cache` |
| [`LLMLoader.from_pretrained`](../saddlellm/LLMLoader.py#L436)<br><sub>`from_pretrained(cls, model_path: str, **kwargs) -> 'LLMLoader'`</sub> | method | 从检查点加载 `LLMLoader`，遵循预训练模型的目录契约。 | `LLMConfig`, `kwargs.pop`, `cls`, `loader.load_model` |
| [`LLMLoader.convert_to_ascend_om`](../saddlellm/LLMLoader.py#L446)<br><sub>`convert_to_ascend_om(self, output_path: str)`</sub> | method | 将模型转换为升腾OM格式 | `warnings.warn`, `ModelConvert`, `convert.execute`, `print`, `RuntimeError` |
| [`main`](../saddlellm/LLMLoader.py#L475)<br><sub>`main()`</sub> | function | 测试案例 | `print`, `LLMLoader`, `loader.load_model`, `loader.generate`, `test_config.get`, `loader.convert_to_ascend_om` |

## `saddlellm/LLModelEvalute.py`

共 15 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`Evaluator.__init__`](../saddlellm/LLModelEvalute.py#L28)<br><sub>`__init__(self, args: Optional[Any]=None, **defaults)`</sub> | method | 初始化 `Evaluator` 实例及其运行依赖。 | — |
| [`Evaluator.evaluate`](../saddlellm/LLModelEvalute.py#L32)<br><sub>`evaluate(self, model_path: str, dataset: str, dataset_config: Optional[str]=None, metrics: Optional[List[str]]=None, max_samples: int=1000, task_type: str='generation', split: str='validation', text_column: Optional[str]=None, target_column: Optional[str]=None, max_length: int=512, max_new_tokens: int=128, batch_size: int=4, cpu: bool=False, output_dir: Optional[str]=None, trust_remote_code: bool=True) -> Dict[str, Any]`</sub> | method | `Evaluator` 中评估`evaluate`的公开操作。 | `torch.cuda.is_available`, `self._load_records`, `ValueError`, `self._load_model`, `self._infer_text_column`, `update`, `self._evaluate_classification`, `self._evaluate_generation`, `os.makedirs`, `os.path.join` |
| [`Evaluator.run`](../saddlellm/LLModelEvalute.py#L96)<br><sub>`run(self) -> Dict[str, Any]`</sub> | method | `Evaluator` 中执行`run`的公开操作。 | `ValueError`, `self.evaluate`, `getattr` |
| [`Evaluator._load_model`](../saddlellm/LLModelEvalute.py#L117)<br><sub>`_load_model(self, model_path: str, task_type: str, device: str, trust_remote_code: bool)`</sub> | method | `Evaluator` 中加载模型的内部辅助逻辑。 | `logger.info`, `AutoTokenizer.from_pretrained`, `to`, `AutoModelForSequenceClassification.from_pretrained`, `load_model_and_tokenizer`, `model.eval` |
| [`Evaluator._load_records`](../saddlellm/LLModelEvalute.py#L142)<br><sub>`_load_records(self, dataset: str, dataset_config: Optional[str], split: str, max_samples: int) -> List[Dict[str, Any]]`</sub> | method | `Evaluator` 中加载`load_records`的内部辅助逻辑。 | `Path`, `path.exists`, `self._load_local_records`, `load_dataset`, `self._dataset_to_records` |
| [`Evaluator._load_local_records`](../saddlellm/LLModelEvalute.py#L160)<br><sub>`_load_local_records(self, path: Path) -> List[Dict[str, Any]]`</sub> | method | `Evaluator` 中加载`load_local_records`的内部辅助逻辑。 | `path.suffix.lower`, `open`, `json.loads`, `line.strip`, `json.load`, `isinstance`, `data.get`, `ValueError` |
| [`Evaluator._dataset_to_records`](../saddlellm/LLModelEvalute.py#L177)<br><sub>`_dataset_to_records(self, ds, max_samples: int) -> List[Dict[str, Any]]`</sub> | method | `Evaluator` 中实现数据集的内部辅助逻辑。 | `hasattr`, `ds.select`, `range`, `min`, `len`, `records.append`, `dict` |
| [`Evaluator._infer_text_column`](../saddlellm/LLModelEvalute.py#L187)<br><sub>`_infer_text_column(self, sample: Dict[str, Any]) -> str`</sub> | method | `Evaluator` 中推断`infer_text_column`的内部辅助逻辑。 | `sample.items`, `isinstance`, `ValueError` |
| [`Evaluator._evaluate_generation`](../saddlellm/LLModelEvalute.py#L196)<br><sub>`_evaluate_generation(self, model, tokenizer, records: List[Dict[str, Any]], text_column: str, target_column: Optional[str], metrics: List[str], device: str, max_length: int, max_new_tokens: int, batch_size: int) -> Dict[str, Dict[str, Any]]`</sub> | method | `Evaluator` 中评估`evaluate_generation`的内部辅助逻辑。 | `str`, `r.get`, `self._compute_perplexity`, `any`, `self._generate_predictions`, `min`, `len`, `results.update`, `self._compute_text_metrics` |
| [`Evaluator._compute_perplexity`](../saddlellm/LLModelEvalute.py#L229)<br><sub>`_compute_perplexity(self, model, tokenizer, texts: List[str], device: str, max_length: int, batch_size: int) -> float`</sub> | method | `Evaluator` 中计算`compute_perplexity`的内部辅助逻辑。 | `range`, `len`, `tokenizer`, `v.to`, `encoded.items`, `clone`, `encoded.get`, `torch.ones_like`, `torch.no_grad`, `model` |
| [`Evaluator._generate_predictions`](../saddlellm/LLModelEvalute.py#L263)<br><sub>`_generate_predictions(self, model, tokenizer, texts: List[str], device: str, max_length: int, max_new_tokens: int) -> List[str]`</sub> | method | `Evaluator` 中生成`generate_predictions`的内部辅助逻辑。 | `to`, `tokenizer`, `torch.no_grad`, `model.generate`, `tokenizer.decode`, `predictions.append`, `strip`, `len`, `generated.strip` |
| [`Evaluator._compute_text_metrics`](../saddlellm/LLModelEvalute.py#L286)<br><sub>`_compute_text_metrics(self, predictions: List[str], references: List[str], metrics: List[str]) -> Dict[str, Dict[str, Any]]`</sub> | method | `Evaluator` 中计算指标的内部辅助逻辑。 | `logger.warning`, `compute`, `evaluate.load`, `rouge.get`, `bleu.get` |
| [`Evaluator._evaluate_classification`](../saddlellm/LLModelEvalute.py#L315)<br><sub>`_evaluate_classification(self, model, tokenizer, records: List[Dict[str, Any]], text_column: str, target_column: str, metrics: List[str], device: str, max_length: int, batch_size: int) -> Dict[str, Dict[str, Any]]`</sub> | method | `Evaluator` 中评估`evaluate_classification`的内部辅助逻辑。 | `pipeline`, `clf`, `str`, `int`, `round` |
| [`parse_args`](../saddlellm/LLModelEvalute.py#L349)<br><sub>`parse_args()`</sub> | function | 模块级解析`parse_args`的公开操作。 | `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args` |
| [`main`](../saddlellm/LLModelEvalute.py#L370)<br><sub>`main()`</sub> | function | 模块级实现`main`的公开操作。 | `logging.basicConfig`, `parse_args`, `logger.setLevel`, `run`, `Evaluator`, `print`, `json.dumps`, `result.get`, `SystemExit` |

## `saddlellm/LatentFlowModel.py`

共 12 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`LatentFlowConfig.__post_init__`](../saddlellm/LatentFlowModel.py#L31)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `LatentFlowConfig` 创建后校验并规范化字段。 | `positive.items`, `int`, `ValueError` |
| [`LatentFlowConfig.num_patches`](../saddlellm/LatentFlowModel.py#L53)<br><sub>`num_patches(self) -> int`</sub> | method | `LatentFlowConfig` 中实现`num_patches`的公开操作。 | — |
| [`LatentFlowConfig.patch_dim`](../saddlellm/LatentFlowModel.py#L59)<br><sub>`patch_dim(self) -> int`</sub> | method | `LatentFlowConfig` 中实现`patch_dim`的公开操作。 | — |
| [`LatentFlowConfig.to_dict`](../saddlellm/LatentFlowModel.py#L62)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `LatentFlowConfig` 转为可序列化字典。 | `asdict` |
| [`_timestep_embedding`](../saddlellm/LatentFlowModel.py#L66)<br><sub>`_timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor`</sub> | function | 模块级实现`timestep_embedding`的内部辅助逻辑。 | `math.log`, `max`, `torch.exp`, `torch.arange`, `timesteps.float`, `torch.cat`, `torch.cos`, `torch.sin`, `F.pad` |
| [`ConditionalLatentFlowTransformer.__init__`](../saddlellm/LatentFlowModel.py#L84)<br><sub>`__init__(self, config: LatentFlowConfig) -> None`</sub> | method | 初始化 `ConditionalLatentFlowTransformer` 实例及其运行依赖。 | `__init__`, `super`, `nn.Conv2d`, `nn.Parameter`, `torch.zeros`, `nn.Sequential`, `nn.LayerNorm`, `nn.Linear`, `nn.SiLU`, `nn.TransformerEncoderLayer` |
| [`ConditionalLatentFlowTransformer._reset_parameters`](../saddlellm/LatentFlowModel.py#L119)<br><sub>`_reset_parameters(self) -> None`</sub> | method | `ConditionalLatentFlowTransformer` 中实现`reset_parameters`的内部辅助逻辑。 | `nn.init.normal_`, `nn.init.zeros_` |
| [`ConditionalLatentFlowTransformer.forward`](../saddlellm/LatentFlowModel.py#L126)<br><sub>`forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `ConditionalLatentFlowTransformer` 的前向计算。 | `self._validate_inputs`, `transpose`, `flatten`, `self.patch_embed`, `to`, `self.time_projection`, `_timestep_embedding`, `self.condition_projection`, `self.position_embedding.to`, `self.transformer` |
| [`ConditionalLatentFlowTransformer.compute_flow_loss`](../saddlellm/LatentFlowModel.py#L148)<br><sub>`compute_flow_loss(self, target_latents: torch.Tensor, condition: torch.Tensor, *, noise: Optional[torch.Tensor]=None, timesteps: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]`</sub> | method | Linear interpolation flow matching from Gaussian noise to data. | `torch.randn_like`, `ValueError`, `torch.rand`, `timesteps.reshape`, `self`, `F.mse_loss`, `predicted_velocity.float`, `target_velocity.float`, `loss.detach`, `torch.sqrt` |
| [`ConditionalLatentFlowTransformer.sample`](../saddlellm/LatentFlowModel.py#L180)<br><sub>`sample(self, condition: torch.Tensor, *, num_steps: int=30, initial_noise: Optional[torch.Tensor]=None) -> torch.Tensor`</sub> | method | Euler-integrate the learned velocity field from noise to data. | `ValueError`, `next`, `self.parameters`, `torch.randn`, `initial_noise.clone`, `self.eval`, `range`, `torch.full`, `self`, `self.train` |
| [`ConditionalLatentFlowTransformer._unpatchify`](../saddlellm/LatentFlowModel.py#L220)<br><sub>`_unpatchify(self, patches: torch.Tensor) -> torch.Tensor`</sub> | method | `ConditionalLatentFlowTransformer` 中实现`unpatchify`的内部辅助逻辑。 | `patches.reshape`, `reshape`, `patches.permute` |
| [`ConditionalLatentFlowTransformer._validate_inputs`](../saddlellm/LatentFlowModel.py#L239)<br><sub>`_validate_inputs(self, latents: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> None`</sub> | method | `ConditionalLatentFlowTransformer` 中校验`validate_inputs`的内部辅助逻辑。 | `tuple`, `ValueError` |

## `saddlellm/LatentFlowTrainer.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`LatentFlowTrainingConfig.__post_init__`](../saddlellm/LatentFlowTrainer.py#L39)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `LatentFlowTrainingConfig` 创建后校验并规范化字段。 | `int`, `getattr`, `ValueError` |
| [`CachedLatentDataset.__init__`](../saddlellm/LatentFlowTrainer.py#L60)<br><sub>`__init__(self, path: str) -> None`</sub> | method | 初始化 `CachedLatentDataset` 实例及其运行依赖。 | `Path`, `source.is_dir`, `source.suffix.lower`, `ShardedNpzStore`, `self._sharded.array_shape`, `source.is_file`, `FileNotFoundError`, `ValueError`, `np.load`, `set` |
| [`CachedLatentDataset.__len__`](../saddlellm/LatentFlowTrainer.py#L96)<br><sub>`__len__(self) -> int`</sub> | method | `CachedLatentDataset` 中实现`len__`的内部辅助逻辑。 | `len` |
| [`CachedLatentDataset.__getitem__`](../saddlellm/LatentFlowTrainer.py#L99)<br><sub>`__getitem__(self, index: int) -> Dict[str, torch.Tensor]`</sub> | method | `CachedLatentDataset` 中实现`getitem__`的内部辅助逻辑。 | `self._sharded.get`, `float`, `torch.from_numpy`, `np.array` |
| [`CachedLatentDataset.infer_model_config`](../saddlellm/LatentFlowTrainer.py#L113)<br><sub>`infer_model_config(self, **overrides: Any) -> LatentFlowConfig`</sub> | method | `CachedLatentDataset` 中推断模型、配置的公开操作。 | `LatentFlowConfig`, `int` |
| [`_resolve_device`](../saddlellm/LatentFlowTrainer.py#L125)<br><sub>`_resolve_device(value: str) -> torch.device`</sub> | function | 模块级解析设备的内部辅助逻辑。 | `torch.device`, `torch.cuda.is_available`, `RuntimeError` |
| [`_autocast`](../saddlellm/LatentFlowTrainer.py#L134)<br><sub>`_autocast(device: torch.device, precision: str)`</sub> | function | 模块级实现`autocast`的内部辅助逻辑。 | `torch.autocast` |
| [`_lr_multiplier`](../saddlellm/LatentFlowTrainer.py#L141)<br><sub>`_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float`</sub> | function | 模块级实现`lr_multiplier`的内部辅助逻辑。 | `max`, `math.cos`, `min` |
| [`save_latent_flow_checkpoint`](../saddlellm/LatentFlowTrainer.py#L148)<br><sub>`save_latent_flow_checkpoint(model: ConditionalLatentFlowTransformer, path: str, *, training: Optional[LatentFlowTrainingConfig]=None, global_step: int=0, optimizer: Optional[torch.optim.Optimizer]=None, scheduler: Optional[torch.optim.lr_scheduler.LambdaLR]=None) -> str`</sub> | function | 模块级保存流程、检查点的公开操作。 | `os.makedirs`, `torch.save`, `model.state_dict`, `os.path.join`, `model.config.to_dict`, `int`, `asdict`, `open`, `json.dump`, `file.write` |
| [`load_latent_flow_checkpoint`](../saddlellm/LatentFlowTrainer.py#L181)<br><sub>`load_latent_flow_checkpoint(path: str, *, map_location: str='cpu') -> ConditionalLatentFlowTransformer`</sub> | function | 模块级加载流程、检查点的公开操作。 | `Path`, `config_path.is_file`, `weights_path.is_file`, `FileNotFoundError`, `json.loads`, `config_path.read_text`, `ConditionalLatentFlowTransformer`, `LatentFlowConfig`, `torch.load`, `model.load_state_dict` |
| [`train_latent_flow`](../saddlellm/LatentFlowTrainer.py#L202)<br><sub>`train_latent_flow(model_config: LatentFlowConfig, training_config: LatentFlowTrainingConfig) -> Dict[str, Any]`</sub> | function | 模块级训练流程的公开操作。 | `random.seed`, `np.random.seed`, `torch.manual_seed`, `CachedLatentDataset`, `ValueError`, `_resolve_device`, `to`, `ConditionalLatentFlowTransformer`, `DataLoader`, `math.ceil` |

## `saddlellm/Lightweight.py`

共 9 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`Lightweight.recommend`](../saddlellm/Lightweight.py#L104)<br><sub>`recommend(model_size: str='1b')`</sub> | method | 查看推荐的轻量化方案。 | `Lightweight._resolve_size`, `LIGHTWEIGHT_RECIPES.get`, `print`, `model_size.upper` |
| [`Lightweight.optimize`](../saddlellm/Lightweight.py#L131)<br><sub>`optimize(model, tokenizer, output_dir: str='./lightweight_model', target: str='auto', apply_quantization: bool=True, apply_pruning: bool=True, apply_kv_cache_quant: bool=True, bits: int=None)`</sub> | method | 一键轻量化 — 自动量化 + 剪枝 + KV Cache 优化。 | `sum`, `p.numel`, `model.parameters`, `Lightweight._resolve_size`, `LIGHTWEIGHT_RECIPES.get`, `print`, `ModelQuantizer`, `q.quantize`, `ModelPruner`, `p.prune` |
| [`Lightweight.quantize_only`](../saddlellm/Lightweight.py#L215)<br><sub>`quantize_only(model, tokenizer, output_dir: str='./quantized_model', bits: int=4, method: str='auto')`</sub> | method | 只做量化。 | `ModelQuantizer`, `q.quantize`, `model.save_pretrained`, `tokenizer.save_pretrained`, `print` |
| [`Lightweight.kv_cache_only`](../saddlellm/Lightweight.py#L226)<br><sub>`kv_cache_only(model, bits: int=8)`</sub> | method | 只优化 KV Cache。 | `Lightweight._configure_kv_cache`, `print` |
| [`Lightweight.speculative_generate`](../saddlellm/Lightweight.py#L236)<br><sub>`speculative_generate(model, tokenizer, prompt: str, draft_model=None, draft_tokenizer=None, max_new_tokens: int=256, temperature: float=0.0, num_draft_tokens: int=5) -> Dict`</sub> | method | 投机解码 — 用小模型打草稿, 大模型审阅。 | `next`, `model.parameters`, `logger.info`, `Lightweight._standard_generate`, `draft_model.parameters`, `model.eval`, `draft_model.eval`, `to`, `tokenizer`, `len` |
| [`Lightweight._standard_generate`](../saddlellm/Lightweight.py#L372)<br><sub>`_standard_generate(model, tokenizer, prompt, max_tokens, temperature)`</sub> | method | `Lightweight` 中生成`standard_generate`的内部辅助逻辑。 | `model.eval`, `to`, `tokenizer`, `torch.no_grad`, `model.generate`, `tokenizer.decode`, `len`, `strip` |
| [`Lightweight._configure_kv_cache`](../saddlellm/Lightweight.py#L388)<br><sub>`_configure_kv_cache(model, bits: int=8)`</sub> | method | 配置 KV Cache 为 int8/int4。 | `logger.info` |
| [`Lightweight._resolve_size`](../saddlellm/Lightweight.py#L413)<br><sub>`_resolve_size(size: str) -> str`</sub> | method | 统一 size 名称。 | `size_map.get`, `size.lower` |
| [`Lightweight.compare_methods`](../saddlellm/Lightweight.py#L425)<br><sub>`compare_methods()`</sub> | method | 轻量化方法对比。 | `print` |

## `saddlellm/LoRATuner.py`

共 6 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`LoRATuner.__init__`](../saddlellm/LoRATuner.py#L24)<br><sub>`__init__(self, model_name: str='meta-llama/Meta-Llama-3-8B', max_length: int=1024, lora_rank: int=8, lora_alpha: int=32, lora_dropout: float=0.05, target_modules: List[str]=['q_proj', 'v_proj'], use_4bit: bool=True, device_map: str='auto')`</sub> | method | Initialize LoRA fine-tuning. :param model_name: Hugging Face model ID :param use_4bit: Enable 4-bit quantization (QLoRA) :param target_modules: Modules to apply LoRA (e.g., ["q_proj", "v_proj"]) | `AutoModelForCausalLM.from_pretrained`, `BitsAndBytesConfig`, `prepare_model_for_kbit_training`, `LoraConfig`, `get_peft_model`, `self.model.print_trainable_parameters`, `AutoTokenizer.from_pretrained` |
| [`LoRATuner.fit`](../saddlellm/LoRATuner.py#L78)<br><sub>`fit(self, train_dataset: Dataset, eval_dataset: Optional[Dataset]=None, epochs: int=3, batch_size: int=2, learning_rate: float=0.0002, output_dir: str='./lora_output', logging_steps: int=10, save_strategy: str='steps', save_steps: int=500, gradient_accumulation_steps: int=4, deepspeed: Optional[str]=None) -> None`</sub> | method | Run LoRA fine-tuning. | `train_dataset.map`, `eval_dataset.map`, `TrainingArguments`, `torch.cuda.is_bf16_supported`, `DataCollatorForLanguageModeling`, `Trainer`, `self.trainer.train` |
| [`LoRATuner.fit.tokenize_fn`](../saddlellm/LoRATuner.py#L95)<br><sub>`tokenize_fn(examples: Dict) -> Dict`</sub> | nested function | `LoRATuner` 中分词`tokenize_fn`的局部回调/辅助逻辑。 | `self.tokenizer` |
| [`LoRATuner.predict`](../saddlellm/LoRATuner.py#L143)<br><sub>`predict(self, text: str, max_new_tokens: int=100) -> str`</sub> | method | Generate text from input prompt. | `to`, `self.tokenizer`, `self.model.generate`, `self.tokenizer.decode` |
| [`LoRATuner.save`](../saddlellm/LoRATuner.py#L159)<br><sub>`save(self, path: str) -> None`</sub> | method | Save only LoRA weights (轻量保存). | `self.model.save_pretrained` |
| [`LoRATuner.load`](../saddlellm/LoRATuner.py#L164)<br><sub>`load(cls, path: str, base_model_name: Optional[str]=None, **kwargs) -> 'LoRATuner'`</sub> | method | Load LoRA weights. :param base_model_name: 基础模型名称（如果未保存tokenizer） | `cls`, `PeftModel.from_pretrained` |

## `saddlellm/MediaCache.py`

共 21 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MediaCacheBuildConfig.__post_init__`](../saddlellm/MediaCache.py#L49)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `MediaCacheBuildConfig` 创建后校验并规范化字段。 | `lower`, `str`, `ValueError` |
| [`ShardedNpzStore.__init__`](../saddlellm/MediaCache.py#L64)<br><sub>`__init__(self, path: str, required_arrays: Sequence[str]) -> None`</sub> | method | 初始化 `ShardedNpzStore` 实例及其运行依赖。 | `Path`, `source.is_dir`, `manifest_path.is_file`, `manifest_path.suffix.lower`, `FileNotFoundError`, `json.loads`, `manifest_path.read_text`, `self.manifest.get`, `ValueError`, `tuple` |
| [`ShardedNpzStore.__len__`](../saddlellm/MediaCache.py#L98)<br><sub>`__len__(self) -> int`</sub> | method | `ShardedNpzStore` 中实现`len__`的内部辅助逻辑。 | — |
| [`ShardedNpzStore.array_shape`](../saddlellm/MediaCache.py#L101)<br><sub>`array_shape(self, name: str) -> Tuple[int, ...]`</sub> | method | `ShardedNpzStore` 中实现`array_shape`的公开操作。 | `tuple`, `int`, `KeyError` |
| [`ShardedNpzStore.array_stat`](../saddlellm/MediaCache.py#L107)<br><sub>`array_stat(self, name: str, statistic: str) -> Any`</sub> | method | `ShardedNpzStore` 中实现`array_stat`的公开操作。 | `KeyError` |
| [`ShardedNpzStore.get`](../saddlellm/MediaCache.py#L115)<br><sub>`get(self, index: int) -> Dict[str, np.ndarray]`</sub> | method | `ShardedNpzStore` 中读取`get`的公开操作。 | `IndexError`, `bisect_right`, `self._open_shard`, `np.array` |
| [`ShardedNpzStore._open_shard`](../saddlellm/MediaCache.py#L128)<br><sub>`_open_shard(self, index: int)`</sub> | method | `ShardedNpzStore` 中实现`open_shard`的内部辅助逻辑。 | `self._cached_archive.close`, `path.is_file`, `FileNotFoundError`, `np.load`, `set`, `ValueError`, `join`, `sorted` |
| [`ShardedNpzStore.close`](../saddlellm/MediaCache.py#L145)<br><sub>`close(self) -> None`</sub> | method | `ShardedNpzStore` 中关闭`close`的公开操作。 | `getattr`, `self._cached_archive.close` |
| [`ShardedNpzStore.__getstate__`](../saddlellm/MediaCache.py#L151)<br><sub>`__getstate__(self) -> Dict[str, Any]`</sub> | method | `ShardedNpzStore` 中实现`getstate__`的内部辅助逻辑。 | `dict` |
| [`ShardedNpzStore.__del__`](../saddlellm/MediaCache.py#L157)<br><sub>`__del__(self) -> None`</sub> | method | `ShardedNpzStore` 中实现`del__`的内部辅助逻辑。 | `self.close` |
| [`build_media_cache`](../saddlellm/MediaCache.py#L161)<br><sub>`build_media_cache(config: MediaCacheBuildConfig \| Dict[str, Any]) -> Dict[str, Any]`</sub> | function | Encode a JSON/JSONL/CSV media manifest into resumable NPZ shards. | `isinstance`, `MediaCacheBuildConfig`, `resolve`, `expanduser`, `Path`, `source.is_file`, `FileNotFoundError`, `output.exists`, `_clear_owned_cache`, `manifest_path.is_file` |
| [`_initial_manifest`](../saddlellm/MediaCache.py#L328)<br><sub>`_initial_manifest(identity: Dict[str, Any], config: MediaCacheBuildConfig) -> Dict[str, Any]`</sub> | function | 模块级实现数据清单的内部辅助逻辑。 | `asdict` |
| [`_validate_resume_identity`](../saddlellm/MediaCache.py#L344)<br><sub>`_validate_resume_identity(previous: Dict[str, Any], identity: Dict[str, Any]) -> None`</sub> | function | 模块级校验`validate_resume_identity`的内部辅助逻辑。 | `identity.items`, `previous.get`, `ValueError` |
| [`_extract_prompt`](../saddlellm/MediaCache.py#L353)<br><sub>`_extract_prompt(record: Dict[str, Any], field: Optional[str]) -> str`</sub> | function | 模块级提取提示词的内部辅助逻辑。 | `record.get`, `next`, `isinstance`, `join`, `str`, `item.get`, `ValueError` |
| [`_extract_media_path`](../saddlellm/MediaCache.py#L376)<br><sub>`_extract_media_path(record: Dict[str, Any], modality: str, field: Optional[str], root: Path) -> Path`</sub> | function | 模块级提取路径的内部辅助逻辑。 | `record.get`, `isinstance`, `lower`, `str`, `item.get`, `value.get`, `ValueError`, `os.fspath`, `text.startswith`, `expanduser` |
| [`_validate_encoded_sample`](../saddlellm/MediaCache.py#L415)<br><sub>`_validate_encoded_sample(modality: str, media: np.ndarray, condition: np.ndarray) -> None`</sub> | function | 模块级校验`validate_encoded_sample`的内部辅助逻辑。 | `ValueError`, `np.issubdtype`, `int`, `media.min`, `all`, `np.isfinite` |
| [`_flush_shard`](../saddlellm/MediaCache.py#L434)<br><sub>`_flush_shard(output: Path, manifest: Dict[str, Any], shard_index: int, array_name: str, media_batch: Sequence[np.ndarray], condition_batch: Sequence[np.ndarray], record_indices: Sequence[int], *, attention_batch: Sequence[np.ndarray], processed_records: int, failures: Sequence[Dict[str, Any]]) -> None`</sub> | function | 模块级实现`flush_shard`的内部辅助逻辑。 | `np.stack`, `astype`, `np.asarray`, `len`, `ValueError`, `np.savez_compressed`, `os.replace`, `_sha256_file`, `int`, `append` |
| [`_file_fingerprint`](../saddlellm/MediaCache.py#L503)<br><sub>`_file_fingerprint(path: Path) -> Dict[str, Any]`</sub> | function | 模块级实现`file_fingerprint`的内部辅助逻辑。 | `path.stat`, `str`, `_sha256_file` |
| [`_sha256_file`](../saddlellm/MediaCache.py#L513)<br><sub>`_sha256_file(path: Path) -> str`</sub> | function | 模块级实现`sha256_file`的内部辅助逻辑。 | `hashlib.sha256`, `path.open`, `stream.read`, `digest.update`, `digest.hexdigest` |
| [`_atomic_write_json`](../saddlellm/MediaCache.py#L524)<br><sub>`_atomic_write_json(path: Path, payload: Dict[str, Any]) -> None`</sub> | function | 模块级写入`atomic_write_json`的内部辅助逻辑。 | `path.with_name`, `temporary.write_text`, `json.dumps`, `os.replace` |
| [`_clear_owned_cache`](../saddlellm/MediaCache.py#L532)<br><sub>`_clear_owned_cache(output: Path) -> None`</sub> | function | Remove only files declared by a SaddleLLM-owned media cache manifest. | `manifest_path.is_file`, `any`, `output.iterdir`, `ValueError`, `json.loads`, `manifest_path.read_text`, `manifest.get`, `resolve`, `output.resolve`, `path.is_file` |

## `saddlellm/ModalityCodec.py`

共 12 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModalityCodecSpec.__post_init__`](../saddlellm/ModalityCodec.py#L20)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `ModalityCodecSpec` 创建后校验并规范化字段。 | `ValueError` |
| [`ModalityCodec.encode`](../saddlellm/ModalityCodec.py#L33)<br><sub>`encode(self, value: Any, **kwargs: Any) -> Any`</sub> | method | `ModalityCodec` 中编码`encode`的公开操作。 | — |
| [`ModalityCodec.decode`](../saddlellm/ModalityCodec.py#L37)<br><sub>`decode(self, latents: Any, **kwargs: Any) -> Any`</sub> | method | `ModalityCodec` 中解码`decode`的公开操作。 | — |
| [`ModalityCodec.fingerprint`](../saddlellm/ModalityCodec.py#L40)<br><sub>`fingerprint(self) -> Dict[str, Any]`</sub> | method | Serializable identity stored beside cached latents/checkpoints. | `dict` |
| [`CallableModalityCodec.__init__`](../saddlellm/ModalityCodec.py#L56)<br><sub>`__init__(self, spec: ModalityCodecSpec, encoder: Callable[..., Any], decoder: Optional[Callable[..., Any]]=None) -> None`</sub> | method | 初始化 `CallableModalityCodec` 实例及其运行依赖。 | `ValueError` |
| [`CallableModalityCodec.encode`](../saddlellm/ModalityCodec.py#L68)<br><sub>`encode(self, value: Any, **kwargs: Any) -> Any`</sub> | method | `CallableModalityCodec` 中编码`encode`的公开操作。 | `self._encoder` |
| [`CallableModalityCodec.decode`](../saddlellm/ModalityCodec.py#L71)<br><sub>`decode(self, latents: Any, **kwargs: Any) -> Any`</sub> | method | `CallableModalityCodec` 中解码`decode`的公开操作。 | `NotImplementedError`, `self._decoder` |
| [`ModalityCodecRegistry.register`](../saddlellm/ModalityCodec.py#L84)<br><sub>`register(cls, spec: ModalityCodecSpec, factory: Callable[..., ModalityCodec], *, overwrite: bool=False) -> None`</sub> | method | `ModalityCodecRegistry` 中注册`register`的公开操作。 | `lower`, `spec.name.strip`, `ValueError` |
| [`ModalityCodecRegistry.build`](../saddlellm/ModalityCodec.py#L100)<br><sub>`build(cls, name: str, **kwargs: Any) -> ModalityCodec`</sub> | method | `ModalityCodecRegistry` 中构建`build`的公开操作。 | `lower`, `strip`, `str`, `KeyError`, `join`, `sorted`, `isinstance`, `TypeError`, `codec.spec.name.lower`, `ValueError` |
| [`ModalityCodecRegistry.get_spec`](../saddlellm/ModalityCodec.py#L118)<br><sub>`get_spec(cls, name: str) -> ModalityCodecSpec`</sub> | method | `ModalityCodecRegistry` 中读取`get_spec`的公开操作。 | `lower`, `strip`, `str`, `KeyError` |
| [`ModalityCodecRegistry.list_specs`](../saddlellm/ModalityCodec.py#L125)<br><sub>`list_specs(cls, modality: Optional[str]=None) -> List[ModalityCodecSpec]`</sub> | method | `ModalityCodecRegistry` 中列出`list_specs`的公开操作。 | `list`, `cls._specs.values`, `modality.lower`, `ValueError`, `sorted` |
| [`ModalityCodecRegistry.unregister`](../saddlellm/ModalityCodec.py#L135)<br><sub>`unregister(cls, name: str) -> None`</sub> | method | Remove a registration; primarily useful for isolated plugin tests. | `lower`, `strip`, `str`, `cls._factories.pop`, `cls._specs.pop` |

## `saddlellm/ModelAdapter.py`

共 8 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelBackendCapabilities.supports_stage`](../saddlellm/ModelAdapter.py#L23)<br><sub>`supports_stage(self, stage: str) -> bool`</sub> | method | `ModelBackendCapabilities` 中实现训练阶段的公开操作。 | — |
| [`ModelAdapter.load_model`](../saddlellm/ModelAdapter.py#L33)<br><sub>`load_model(self, path: str, **kwargs: Any) -> Any`</sub> | method | `ModelAdapter` 中加载模型的公开操作。 | `load_causal_lm` |
| [`ModelAdapter.load_tokenizer`](../saddlellm/ModelAdapter.py#L38)<br><sub>`load_tokenizer(self, path: str, **kwargs: Any) -> Any`</sub> | method | `ModelAdapter` 中加载分词器的公开操作。 | `load_tokenizer_compatible` |
| [`ModelAdapter.trainer_class`](../saddlellm/ModelAdapter.py#L43)<br><sub>`trainer_class(self)`</sub> | method | `ModelAdapter` 中实现`trainer_class`的公开操作。 | — |
| [`ModelAdapter.validate_stage`](../saddlellm/ModelAdapter.py#L48)<br><sub>`validate_stage(self, stage: str, *, method: Optional[str]=None, use_lora: bool=False, use_qlora: bool=False) -> None`</sub> | method | `ModelAdapter` 中校验训练阶段的公开操作。 | `self.capabilities.supports_stage`, `join`, `sorted`, `ValueError` |
| [`SaddleModelAdapter.load_model`](../saddlellm/ModelAdapter.py#L104)<br><sub>`load_model(self, path: str, **kwargs: Any) -> Any`</sub> | method | `SaddleModelAdapter` 中加载模型的公开操作。 | `is_saddle_checkpoint`, `ValueError`, `load_model`, `super` |
| [`SaddleModelAdapter.trainer_class`](../saddlellm/ModelAdapter.py#L114)<br><sub>`trainer_class(self)`</sub> | method | `SaddleModelAdapter` 中实现`trainer_class`的公开操作。 | — |
| [`get_model_adapter`](../saddlellm/ModelAdapter.py#L155)<br><sub>`get_model_adapter(backend: str) -> ModelAdapter`</sub> | function | Resolve a configured backend without inspecting a remote model ID. | `lower`, `strip`, `str`, `ValueError`, `join`, `sorted` |

## `saddlellm/ModelBlueprint.py`

共 46 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`AttentionBlueprint.__post_init__`](../saddlellm/ModelBlueprint.py#L29)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `AttentionBlueprint` 创建后校验并规范化字段。 | `lower`, `str` |
| [`FFNBlueprint.__post_init__`](../saddlellm/ModelBlueprint.py#L48)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `FFNBlueprint` 创建后校验并规范化字段。 | `lower`, `str` |
| [`ResidualBlueprint.__post_init__`](../saddlellm/ModelBlueprint.py#L69)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `ResidualBlueprint` 创建后校验并规范化字段。 | `replace`, `lower`, `str`, `get` |
| [`LayerBlueprint.__post_init__`](../saddlellm/ModelBlueprint.py#L95)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `LayerBlueprint` 创建后校验并规范化字段。 | `isinstance`, `TypeError`, `ValueError` |
| [`LayerBlueprint.to_dict`](../saddlellm/ModelBlueprint.py#L101)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `LayerBlueprint` 转为可序列化字典。 | `component_dict` |
| [`LayerBlueprint.to_dict.component_dict`](../saddlellm/ModelBlueprint.py#L102)<br><sub>`component_dict(value)`</sub> | nested function | `LayerBlueprint` 中实现`component_dict`的局部回调/辅助逻辑。 | `isinstance`, `dict`, `asdict` |
| [`ModelBlueprint.__post_init__`](../saddlellm/ModelBlueprint.py#L167)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `ModelBlueprint` 创建后校验并规范化字段。 | `isinstance`, `AttentionBlueprint`, `FFNBlueprint`, `self._residual_from_dict`, `ObjectiveBlueprint`, `VisionBlueprint`, `ProjectorBlueprint`, `TypeError`, `self._layer_from_dict`, `item.to_dict` |
| [`ModelBlueprint.from_dict`](../saddlellm/ModelBlueprint.py#L220)<br><sub>`from_dict(cls, data: Dict[str, Any], defaults: Optional['ModelBlueprint']=None) -> 'ModelBlueprint'`</sub> | method | Create a blueprint from JSON/YAML-compatible data. | `dict`, `isinstance`, `raw.get`, `get`, `raw.pop`, `cls._reject_unknown_keys`, `defaults.to_config_dict`, `cls._ensure_mapping`, `AttentionBlueprint`, `asdict` |
| [`ModelBlueprint.from_file`](../saddlellm/ModelBlueprint.py#L297)<br><sub>`from_file(cls, path: str) -> 'ModelBlueprint'`</sub> | method | Load a standalone blueprint or a training config containing one. | `os.fspath`, `open`, `endswith`, `path_string.lower`, `yaml.safe_load`, `json.load`, `isinstance`, `ValueError`, `cls.from_dict` |
| [`ModelBlueprint._layer_from_dict`](../saddlellm/ModelBlueprint.py#L313)<br><sub>`_layer_from_dict(data: Dict[str, Any], default_attention: AttentionBlueprint, default_ffn: FFNBlueprint, default_residual: ResidualBlueprint) -> LayerBlueprint`</sub> | method | `ModelBlueprint` 中实现`layer_from_dict`的内部辅助逻辑。 | `dict`, `ModelBlueprint._reject_unknown_keys`, `ModelBlueprint._ensure_mapping`, `raw.get`, `isinstance`, `TypeError`, `AttentionBlueprint`, `asdict`, `FFNBlueprint`, `ResidualBlueprint` |
| [`ModelBlueprint._residual_from_dict`](../saddlellm/ModelBlueprint.py#L352)<br><sub>`_residual_from_dict(data: Dict[str, Any]) -> ResidualBlueprint`</sub> | method | `ModelBlueprint` 中实现`residual_from_dict`的内部辅助逻辑。 | `ResidualBlueprint`, `ModelBlueprint._residual_values` |
| [`ModelBlueprint._residual_values`](../saddlellm/ModelBlueprint.py#L356)<br><sub>`_residual_values(data: Dict[str, Any]) -> Dict[str, Any]`</sub> | method | `ModelBlueprint` 中实现`residual_values`的内部辅助逻辑。 | `dict`, `ModelBlueprint._consume_alias`, `values.pop`, `lower`, `str`, `ValueError`, `values.setdefault`, `isinstance`, `sorted`, `set` |
| [`ModelBlueprint._ensure_mapping`](../saddlellm/ModelBlueprint.py#L395)<br><sub>`_ensure_mapping(value: Any, label: str) -> None`</sub> | method | `ModelBlueprint` 中确保`ensure_mapping`的内部辅助逻辑。 | `isinstance`, `TypeError` |
| [`ModelBlueprint._consume_alias`](../saddlellm/ModelBlueprint.py#L400)<br><sub>`_consume_alias(values: Dict[str, Any], alias: str, canonical: str) -> None`</sub> | method | `ModelBlueprint` 中实现`consume_alias`的内部辅助逻辑。 | `values.pop`, `ValueError`, `values.setdefault` |
| [`ModelBlueprint._reject_unknown_keys`](../saddlellm/ModelBlueprint.py#L409)<br><sub>`_reject_unknown_keys(data: Dict[str, Any], component, label: str) -> None`</sub> | method | `ModelBlueprint` 中实现`reject_unknown_keys`的内部辅助逻辑。 | `fields`, `sorted`, `set`, `ValueError`, `join` |
| [`ModelBlueprint.validate`](../saddlellm/ModelBlueprint.py#L415)<br><sub>`validate(self) -> None`</sub> | method | `ModelBlueprint` 中校验`validate`的公开操作。 | `ValueError`, `lower`, `str`, `enumerate`, `self.expanded_layers`, `isinstance`, `TypeError`, `attention.rope_scaling.get`, `math.isfinite` |
| [`ModelBlueprint.expanded_layers`](../saddlellm/ModelBlueprint.py#L549)<br><sub>`expanded_layers(self) -> List[LayerBlueprint]`</sub> | method | Return one fully resolved blueprint per decoder layer. | `LayerBlueprint`, `AttentionBlueprint`, `asdict`, `FFNBlueprint`, `ResidualBlueprint`, `range`, `expanded.append`, `len` |
| [`ModelBlueprint.to_config_dict`](../saddlellm/ModelBlueprint.py#L577)<br><sub>`to_config_dict(self) -> Dict[str, Any]`</sub> | method | Return only the serializable architecture source of truth. | `asdict`, `layer.to_dict` |
| [`ModelBlueprint.to_dict`](../saddlellm/ModelBlueprint.py#L584)<br><sub>`to_dict(self, include_analysis: bool=True) -> Dict`</sub> | method | 把 `ModelBlueprint` 转为可序列化字典。 | `self.to_config_dict`, `self.analyze` |
| [`ModelBlueprint.save`](../saddlellm/ModelBlueprint.py#L590)<br><sub>`save(self, path: str) -> str`</sub> | method | `ModelBlueprint` 中保存`save`的公开操作。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump`, `self.to_dict` |
| [`ModelBlueprint.analyze`](../saddlellm/ModelBlueprint.py#L596)<br><sub>`analyze(self) -> Dict`</sub> | method | `ModelBlueprint` 中分析`analyze`的公开操作。 | `self.estimate_total_params`, `self.estimate_active_params`, `self.cost_profile`, `self._human_number`, `self.estimate_kv_cache_bytes_per_token`, `self.architecture_axes`, `self.risk_report`, `self.training_hypotheses`, `self.multimodal_analysis` |
| [`ModelBlueprint.to_model_spec`](../saddlellm/ModelBlueprint.py#L615)<br><sub>`to_model_spec(self)`</sub> | method | Convert to the existing ModelSpec format for scratch experiments. | `ModelSpec` |
| [`ModelBlueprint.to_transformers_config_dict`](../saddlellm/ModelBlueprint.py#L645)<br><sub>`to_transformers_config_dict(self) -> Dict`</sub> | method | Return a runnable HF-style config dict for dense fallback models. | `self.family.startswith` |
| [`ModelBlueprint.to_saddle_config`](../saddlellm/ModelBlueprint.py#L672)<br><sub>`to_saddle_config(self)`</sub> | method | Convert this blueprint into SaddleLLM's modular model config. | `SaddleModelConfig.from_blueprint` |
| [`ModelBlueprint.build_model`](../saddlellm/ModelBlueprint.py#L678)<br><sub>`build_model(self)`</sub> | method | Instantiate a modular PyTorch SaddleForCausalLM. | `SaddleForCausalLM.from_blueprint` |
| [`ModelBlueprint.is_multimodal`](../saddlellm/ModelBlueprint.py#L684)<br><sub>`is_multimodal(self) -> bool`</sub> | method | `ModelBlueprint` 中实现`is_multimodal`的公开操作。 | `bool` |
| [`ModelBlueprint.build_multimodal_model`](../saddlellm/ModelBlueprint.py#L687)<br><sub>`build_multimodal_model(self, image_token_id: Optional[int]=None)`</sub> | method | Instantiate a LLaVA-style multimodal model. | `VisionBackboneConfig`, `MultimodalProjectorConfig`, `MultimodalForCausalLM.from_text_blueprint` |
| [`ModelBlueprint.estimate_total_params`](../saddlellm/ModelBlueprint.py#L717)<br><sub>`estimate_total_params(self) -> int`</sub> | method | `ModelBlueprint` 中估算`estimate_total_params`的公开操作。 | `self._estimate_parameter_counts` |
| [`ModelBlueprint.estimate_active_params`](../saddlellm/ModelBlueprint.py#L721)<br><sub>`estimate_active_params(self) -> int`</sub> | method | `ModelBlueprint` 中估算`estimate_active_params`的公开操作。 | `self._estimate_parameter_counts` |
| [`ModelBlueprint._estimate_parameter_counts`](../saddlellm/ModelBlueprint.py#L725)<br><sub>`_estimate_parameter_counts(self) -> tuple[int, int]`</sub> | method | Estimate heterogeneous decoder parameters and active parameters. | `self.expanded_layers`, `max`, `len`, `range` |
| [`ModelBlueprint.estimate_kv_cache_bytes_per_token`](../saddlellm/ModelBlueprint.py#L772)<br><sub>`estimate_kv_cache_bytes_per_token(self, dtype_bytes: int=2) -> int`</sub> | method | `ModelBlueprint` 中估算缓存的公开操作。 | `self.expanded_layers`, `max` |
| [`ModelBlueprint.estimate_training_flops`](../saddlellm/ModelBlueprint.py#L786)<br><sub>`estimate_training_flops(self, token_budget: int) -> float`</sub> | method | Rough dense-equivalent training FLOPs for architecture comparisons. | `min`, `float`, `self.estimate_active_params` |
| [`ModelBlueprint.estimate_kv_cache_gb`](../saddlellm/ModelBlueprint.py#L793)<br><sub>`estimate_kv_cache_gb(self, batch_size: int=1, sequence_length: Optional[int]=None, dtype_bytes: int=2) -> float`</sub> | method | `ModelBlueprint` 中估算缓存的公开操作。 | `self.estimate_kv_cache_bytes_per_token`, `max`, `int` |
| [`ModelBlueprint.cost_profile`](../saddlellm/ModelBlueprint.py#L804)<br><sub>`cost_profile(self, token_budget: Optional[int]=None, batch_size: int=1, sequence_length: Optional[int]=None, dtype_bytes: int=2) -> Dict`</sub> | method | `ModelBlueprint` 中实现`cost_profile`的公开操作。 | `int`, `self.estimate_active_params`, `self.estimate_training_flops`, `self.estimate_kv_cache_gb`, `self.estimate_total_params`, `max`, `round` |
| [`ModelBlueprint.architecture_axes`](../saddlellm/ModelBlueprint.py#L835)<br><sub>`architecture_axes(self) -> Dict`</sub> | method | `ModelBlueprint` 中实现`architecture_axes`的公开操作。 | `self.expanded_layers`, `list`, `dict.fromkeys`, `len` |
| [`ModelBlueprint.multimodal_analysis`](../saddlellm/ModelBlueprint.py#L855)<br><sub>`multimodal_analysis(self) -> Dict`</sub> | method | `ModelBlueprint` 中实现`multimodal_analysis`的公开操作。 | `max` |
| [`ModelBlueprint.risk_report`](../saddlellm/ModelBlueprint.py#L868)<br><sub>`risk_report(self) -> List[str]`</sub> | method | `ModelBlueprint` 中报告报告的公开操作。 | `self.expanded_layers`, `risks.append` |
| [`ModelBlueprint.training_hypotheses`](../saddlellm/ModelBlueprint.py#L889)<br><sub>`training_hypotheses(self) -> List[str]`</sub> | method | `ModelBlueprint` 中实现训练的公开操作。 | `self.expanded_layers`, `ideas.append` |
| [`ModelBlueprint._human_number`](../saddlellm/ModelBlueprint.py#L913)<br><sub>`_human_number(value: int) -> str`</sub> | method | `ModelBlueprint` 中实现`human_number`的内部辅助逻辑。 | `str` |
| [`ModelBlueprintLab.dense_gqa`](../saddlellm/ModelBlueprint.py#L925)<br><sub>`dense_gqa(name: str='saddle-dense-gqa-300m', hidden_size: int=1024, layers: int=24, heads: int=16, kv_heads: int=4, vocab_size: int=32000, seq_length: int=4096) -> ModelBlueprint`</sub> | method | `ModelBlueprintLab` 中实现`dense_gqa`的公开操作。 | `ModelBlueprint`, `AttentionBlueprint`, `FFNBlueprint` |
| [`ModelBlueprintLab.deepseek_style_moe`](../saddlellm/ModelBlueprint.py#L947)<br><sub>`deepseek_style_moe(name: str='saddle-deepseek-style-moe', hidden_size: int=1536, layers: int=24, heads: int=24, kv_heads: int=4, experts: int=16, experts_per_token: int=2, seq_length: int=8192) -> ModelBlueprint`</sub> | method | `ModelBlueprintLab` 中实现`deepseek_style_moe`的公开操作。 | `ModelBlueprint`, `AttentionBlueprint`, `max`, `FFNBlueprint`, `ObjectiveBlueprint` |
| [`ModelBlueprintLab.minimax_style_long_context`](../saddlellm/ModelBlueprint.py#L986)<br><sub>`minimax_style_long_context(name: str='saddle-minimax-style-long-context', hidden_size: int=1024, layers: int=24, heads: int=16, seq_length: int=65536) -> ModelBlueprint`</sub> | method | `ModelBlueprintLab` 中实现`minimax_style_long_context`的公开操作。 | `ModelBlueprint`, `AttentionBlueprint`, `max`, `FFNBlueprint` |
| [`ModelBlueprintLab.glm_style_reasoning`](../saddlellm/ModelBlueprint.py#L1011)<br><sub>`glm_style_reasoning(name: str='saddle-glm-style-reasoning', hidden_size: int=1024, layers: int=24, heads: int=16, seq_length: int=8192) -> ModelBlueprint`</sub> | method | `ModelBlueprintLab` 中实现`glm_style_reasoning`的公开操作。 | `ModelBlueprint`, `AttentionBlueprint`, `max`, `FFNBlueprint`, `ObjectiveBlueprint` |
| [`ModelBlueprintLab.llava_style_vlm`](../saddlellm/ModelBlueprint.py#L1034)<br><sub>`llava_style_vlm(name: str='saddle-llava-style-vlm', hidden_size: int=1024, layers: int=24, heads: int=16, kv_heads: int=4, vocab_size: int=32000, seq_length: int=4096, vision_backbone: str='tiny_patch', projector: str='mlp', image_tokens: int=64) -> ModelBlueprint`</sub> | method | `ModelBlueprintLab` 中实现`llava_style_vlm`的公开操作。 | `ModelBlueprint`, `AttentionBlueprint`, `FFNBlueprint`, `VisionBlueprint`, `ProjectorBlueprint`, `ObjectiveBlueprint` |
| [`ModelBlueprintLab.compare`](../saddlellm/ModelBlueprint.py#L1068)<br><sub>`compare(blueprints: List[ModelBlueprint]) -> Dict`</sub> | method | `ModelBlueprintLab` 中比较`compare`的公开操作。 | `bp.analyze` |
| [`ModelBlueprintLab.save_comparison`](../saddlellm/ModelBlueprint.py#L1081)<br><sub>`save_comparison(blueprints: List[ModelBlueprint], path: str) -> str`</sub> | method | `ModelBlueprintLab` 中保存`save_comparison`的公开操作。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump`, `ModelBlueprintLab.compare` |

## `saddlellm/ModelBuilder.py`

共 24 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelComponentRegistry.__init__`](../saddlellm/ModelBuilder.py#L81)<br><sub>`__init__(self, include_builtins: bool=True) -> None`</sub> | method | 初始化 `ModelComponentRegistry` 实例及其运行依赖。 | `self._register_builtins` |
| [`ModelComponentRegistry._normalize_name`](../saddlellm/ModelBuilder.py#L89)<br><sub>`_normalize_name(name: str, label: str='preset') -> str`</sub> | method | `ModelComponentRegistry` 中规范化`normalize_name`的内部辅助逻辑。 | `lower`, `strip`, `str`, `ValueError` |
| [`ModelComponentRegistry._normalize_component`](../saddlellm/ModelBuilder.py#L96)<br><sub>`_normalize_component(cls, component: str) -> str`</sub> | method | `ModelComponentRegistry` 中规范化`normalize_component`的内部辅助逻辑。 | `lower`, `strip`, `str`, `join`, `sorted`, `ValueError` |
| [`ModelComponentRegistry.register`](../saddlellm/ModelBuilder.py#L105)<br><sub>`register(self, component: str, name: str, preset: PresetValue, *, overwrite: bool=False) -> 'ModelComponentRegistry'`</sub> | method | Register one component preset and return the registry for chaining. | `self._normalize_component`, `self._normalize_name`, `callable`, `isinstance`, `TypeError`, `ValueError` |
| [`ModelComponentRegistry.register_attention`](../saddlellm/ModelBuilder.py#L131)<br><sub>`register_attention(self, name: str, preset: PresetValue, *, overwrite: bool=False) -> 'ModelComponentRegistry'`</sub> | method | `ModelComponentRegistry` 中注册注意力的公开操作。 | `self.register` |
| [`ModelComponentRegistry.register_ffn`](../saddlellm/ModelBuilder.py#L140)<br><sub>`register_ffn(self, name: str, preset: PresetValue, *, overwrite: bool=False) -> 'ModelComponentRegistry'`</sub> | method | `ModelComponentRegistry` 中注册`register_ffn`的公开操作。 | `self.register` |
| [`ModelComponentRegistry.register_residual`](../saddlellm/ModelBuilder.py#L149)<br><sub>`register_residual(self, name: str, preset: PresetValue, *, overwrite: bool=False) -> 'ModelComponentRegistry'`</sub> | method | `ModelComponentRegistry` 中注册`register_residual`的公开操作。 | `self.register` |
| [`ModelComponentRegistry.create`](../saddlellm/ModelBuilder.py#L158)<br><sub>`create(self, component: str, name: str, **overrides: Any) -> ComponentBlueprint`</sub> | method | Create a fresh component from a named preset plus field overrides. | `self._normalize_component`, `self._normalize_name`, `join`, `self.available`, `KeyError`, `callable`, `preset`, `isinstance`, `deepcopy`, `dict` |
| [`ModelComponentRegistry.create_attention`](../saddlellm/ModelBuilder.py#L184)<br><sub>`create_attention(self, name: str, **overrides: Any) -> AttentionBlueprint`</sub> | method | `ModelComponentRegistry` 中创建注意力的公开操作。 | `self.create` |
| [`ModelComponentRegistry.create_ffn`](../saddlellm/ModelBuilder.py#L187)<br><sub>`create_ffn(self, name: str, **overrides: Any) -> FFNBlueprint`</sub> | method | `ModelComponentRegistry` 中创建`create_ffn`的公开操作。 | `self.create` |
| [`ModelComponentRegistry.create_residual`](../saddlellm/ModelBuilder.py#L190)<br><sub>`create_residual(self, name: str, **overrides: Any) -> ResidualBlueprint`</sub> | method | `ModelComponentRegistry` 中创建`create_residual`的公开操作。 | `self.create` |
| [`ModelComponentRegistry.available`](../saddlellm/ModelBuilder.py#L193)<br><sub>`available(self, component: Optional[str]=None) -> Union[List[str], Dict[str, List[str]]]`</sub> | method | List preset names for one component, or all components. | `self._normalize_component`, `sorted`, `self._presets.items` |
| [`ModelComponentRegistry._construct`](../saddlellm/ModelBuilder.py#L205)<br><sub>`_construct(cls, component: str, values: Mapping[str, Any]) -> ComponentBlueprint`</sub> | method | `ModelComponentRegistry` 中实现`construct`的内部辅助逻辑。 | `deepcopy`, `dict`, `ModelBlueprint._residual_from_dict`, `component_type` |
| [`ModelComponentRegistry._register_builtins`](../saddlellm/ModelBuilder.py#L216)<br><sub>`_register_builtins(self) -> None`</sub> | method | `ModelComponentRegistry` 中注册`register_builtins`的内部辅助逻辑。 | `attention_presets.items`, `self.register_attention`, `ffn_presets.items`, `self.register_ffn`, `residual_presets.items`, `self.register_residual` |
| [`SaddleModelBuilder.__init__`](../saddlellm/ModelBuilder.py#L270)<br><sub>`__init__(self, name: str, *, family: str='llama', hidden_size: int=1024, num_layers: int=24, vocab_size: int=32000, max_position_embeddings: int=4096, registry: Optional[ModelComponentRegistry]=None, **model_options: Any) -> None`</sub> | method | 初始化 `SaddleModelBuilder` 实例及其运行依赖。 | `ModelComponentRegistry`, `AttentionBlueprint`, `FFNBlueprint`, `ResidualBlueprint`, `ObjectiveBlueprint`, `self.configure` |
| [`SaddleModelBuilder.configure`](../saddlellm/ModelBuilder.py#L299)<br><sub>`configure(self, **model_options: Any) -> 'SaddleModelBuilder'`</sub> | method | Set model-level blueprint fields such as norm or initializer. | `set`, `TypeError`, `join`, `sorted`, `self._model_options.update`, `deepcopy` |
| [`SaddleModelBuilder.defaults`](../saddlellm/ModelBuilder.py#L314)<br><sub>`defaults(self, *, attention: Optional[ComponentInput]=None, ffn: Optional[ComponentInput]=None, residual: Optional[ComponentInput]=None) -> 'SaddleModelBuilder'`</sub> | method | Set components inherited by layer segments that omit overrides. | `self._resolve_component` |
| [`SaddleModelBuilder.add_layers`](../saddlellm/ModelBuilder.py#L342)<br><sub>`add_layers(self, repeat: int=1, *, name: str='', attention: Optional[ComponentInput]=None, ffn: Optional[ComponentInput]=None, residual: Optional[ComponentInput]=None) -> 'SaddleModelBuilder'`</sub> | method | Append a repeatable decoder-layer segment. | `isinstance`, `ValueError`, `self._resolve_component`, `self._layers.append`, `LayerBlueprint`, `str` |
| [`SaddleModelBuilder.add_layer`](../saddlellm/ModelBuilder.py#L381)<br><sub>`add_layer(self, *, name: str='', attention: Optional[ComponentInput]=None, ffn: Optional[ComponentInput]=None, residual: Optional[ComponentInput]=None) -> 'SaddleModelBuilder'`</sub> | method | Convenience form of :meth:`add_layers` with ``repeat=1``. | `self.add_layers` |
| [`SaddleModelBuilder.objective`](../saddlellm/ModelBuilder.py#L399)<br><sub>`objective(self, value: Optional[Union[ObjectiveBlueprint, Mapping[str, Any]]]=None, **overrides: Any) -> 'SaddleModelBuilder'`</sub> | method | Set or update the training objective blueprint. | `asdict`, `isinstance`, `deepcopy`, `dict`, `TypeError`, `payload.update`, `ObjectiveBlueprint` |
| [`SaddleModelBuilder.build_blueprint`](../saddlellm/ModelBuilder.py#L418)<br><sub>`build_blueprint(self) -> ModelBlueprint`</sub> | method | Build and validate the serializable architecture blueprint. | `deepcopy`, `payload.update`, `asdict`, `layer.to_dict`, `ModelBlueprint.from_dict` |
| [`SaddleModelBuilder.build`](../saddlellm/ModelBuilder.py#L435)<br><sub>`build(self)`</sub> | method | Instantiate :class:`SaddleForCausalLM` from the validated blueprint. | `build_model`, `self.build_blueprint` |
| [`SaddleModelBuilder.to_dict`](../saddlellm/ModelBuilder.py#L440)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | Return the same configuration mapping accepted by file-based use. | `to_config_dict`, `self.build_blueprint` |
| [`SaddleModelBuilder._resolve_component`](../saddlellm/ModelBuilder.py#L445)<br><sub>`_resolve_component(self, component: str, value: ComponentInput, *, base: Optional[ComponentBlueprint]=None) -> ComponentBlueprint`</sub> | method | `SaddleModelBuilder` 中解析`resolve_component`的内部辅助逻辑。 | `isinstance`, `self.registry.create`, `ModelComponentRegistry._construct`, `asdict`, `TypeError`, `deepcopy`, `dict`, `payload.pop`, `str`, `ModelBlueprint._residual_values` |

## `saddlellm/ModelExperimentPlanner.py`

共 14 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelExperimentConfig.to_dict`](../saddlellm/ModelExperimentPlanner.py#L32)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ModelExperimentConfig` 转为可序列化字典。 | `asdict` |
| [`ModelExperimentRun.to_dict`](../saddlellm/ModelExperimentPlanner.py#L50)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ModelExperimentRun` 转为可序列化字典。 | `asdict` |
| [`ModelExperimentBundle.to_dict`](../saddlellm/ModelExperimentPlanner.py#L65)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ModelExperimentBundle` 转为可序列化字典。 | `run.to_dict` |
| [`ModelExperimentPlanner.__init__`](../saddlellm/ModelExperimentPlanner.py#L79)<br><sub>`__init__(self, config: Optional[ModelExperimentConfig]=None, blueprints: Optional[Sequence]=None)`</sub> | method | 初始化 `ModelExperimentPlanner` 实例及其运行依赖。 | `ModelExperimentConfig`, `list`, `self.default_blueprints` |
| [`ModelExperimentPlanner.default_blueprints`](../saddlellm/ModelExperimentPlanner.py#L83)<br><sub>`default_blueprints(self)`</sub> | method | `ModelExperimentPlanner` 中实现`default_blueprints`的公开操作。 | `ModelBlueprintLab.dense_gqa`, `blueprints.append`, `ModelBlueprint`, `AttentionBlueprint`, `FFNBlueprint`, `ObjectiveBlueprint`, `ModelBlueprintLab.minimax_style_long_context`, `max` |
| [`ModelExperimentPlanner.build_bundle`](../saddlellm/ModelExperimentPlanner.py#L169)<br><sub>`build_bundle(self) -> ModelExperimentBundle`</sub> | method | `ModelExperimentPlanner` 中构建`build_bundle`的公开操作。 | `os.makedirs`, `self._save_json`, `os.path.join`, `self.config.to_dict`, `ModelBlueprintLab.save_comparison`, `enumerate`, `bp.save`, `self._recipe_for_blueprint`, `recipe.save`, `recipe.compile` |
| [`ModelExperimentPlanner.run`](../saddlellm/ModelExperimentPlanner.py#L255)<br><sub>`run(self, dry_run: bool=True, limit: Optional[int]=None) -> ModelExperimentBundle`</sub> | method | `ModelExperimentPlanner` 中执行`run`的公开操作。 | `self.build_bundle`, `len`, `open`, `json.load`, `get`, `cfg.get`, `run`, `TrainingOrchestrator.from_dict`, `PretrainReport.generate`, `self._save_json` |
| [`ModelExperimentPlanner._recipe_for_blueprint`](../saddlellm/ModelExperimentPlanner.py#L274)<br><sub>`_recipe_for_blueprint(self, bp, run_dir: str)`</sub> | method | `ModelExperimentPlanner` 中实现训练配方、模型蓝图的内部辅助逻辑。 | `TrainingRecipe`, `RecipeModelConfig`, `bp.estimate_total_params`, `get`, `bp.architecture_axes`, `RecipeDataConfig`, `list`, `os.path.join`, `RecipeBackendConfig`, `RecipeTrainingConfig` |
| [`ModelExperimentPlanner._experiment_token_budget`](../saddlellm/ModelExperimentPlanner.py#L321)<br><sub>`_experiment_token_budget(self) -> int`</sub> | method | `ModelExperimentPlanner` 中实现`experiment_token_budget`的内部辅助逻辑。 | `int`, `max` |
| [`ModelExperimentPlanner._score_blueprint`](../saddlellm/ModelExperimentPlanner.py#L324)<br><sub>`_score_blueprint(self, bp, analysis: Dict, backend_plan: Dict) -> Dict`</sub> | method | `ModelExperimentPlanner` 中评分模型蓝图的内部辅助逻辑。 | `analysis.get`, `len`, `backend_plan.get`, `axes.get`, `min`, `cost.get`, `max`, `round` |
| [`ModelExperimentPlanner._build_summary`](../saddlellm/ModelExperimentPlanner.py#L377)<br><sub>`_build_summary(self, runs: List[ModelExperimentRun]) -> Dict`</sub> | method | `ModelExperimentPlanner` 中构建`build_summary`的内部辅助逻辑。 | `sorted`, `run.score.get`, `get`, `run.analysis.get`, `run.backend_plan.get`, `warnings.append`, `self._experiment_token_budget`, `len`, `self._summary_item`, `list_attention_backends` |
| [`ModelExperimentPlanner._summary_item`](../saddlellm/ModelExperimentPlanner.py#L403)<br><sub>`_summary_item(self, run: ModelExperimentRun) -> Dict`</sub> | method | `ModelExperimentPlanner` 中实现`summary_item`的内部辅助逻辑。 | `run.analysis.get`, `run.score.get`, `round`, `cost.get` |
| [`ModelExperimentPlanner._next_actions`](../saddlellm/ModelExperimentPlanner.py#L417)<br><sub>`_next_actions(self, smoke_queue: List[ModelExperimentRun], research_queue: List[ModelExperimentRun]) -> List[str]`</sub> | method | `ModelExperimentPlanner` 中实现`next_actions`的内部辅助逻辑。 | `actions.append`, `len` |
| [`ModelExperimentPlanner._save_json`](../saddlellm/ModelExperimentPlanner.py#L428)<br><sub>`_save_json(self, path: str, data: Dict) -> str`</sub> | method | `ModelExperimentPlanner` 中保存`save_json`的内部辅助逻辑。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump` |

## `saddlellm/ModelExporter.py`

共 22 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelExportResult.to_dict`](../saddlellm/ModelExporter.py#L64)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `ModelExportResult` 转为可序列化字典。 | `asdict` |
| [`ModelExporter.export`](../saddlellm/ModelExporter.py#L78)<br><sub>`export(cls, request: ModelExportRequest, *, gate: Optional[Mapping[str, Any]]=None) -> ModelExportResult`</sub> | method | `ModelExporter` 中导出`export`的公开操作。 | `cls._validate_request`, `cls._canonical_format`, `resolve`, `expanduser`, `Path`, `source_path.exists`, `source_path.resolve`, `cls._validate_path_relationship`, `cls._is_native_checkpoint`, `cls._validate_source_format` |
| [`ModelExporter._validate_request`](../saddlellm/ModelExporter.py#L195)<br><sub>`_validate_request(request: ModelExportRequest) -> None`</sub> | method | `ModelExporter` 中校验请求的内部辅助逻辑。 | `strip`, `str`, `ValueError`, `ModelExporter._canonical_format` |
| [`ModelExporter._canonical_format`](../saddlellm/ModelExporter.py#L207)<br><sub>`_canonical_format(format: str) -> str`</sub> | method | `ModelExporter` 中格式化`canonical_format`的内部辅助逻辑。 | `replace`, `lower`, `strip`, `str`, `ValueError` |
| [`ModelExporter._is_native_checkpoint`](../saddlellm/ModelExporter.py#L219)<br><sub>`_is_native_checkpoint(source: Path) -> bool`</sub> | method | `ModelExporter` 中实现检查点的内部辅助逻辑。 | `source.is_dir`, `is_file` |
| [`ModelExporter._validate_source_format`](../saddlellm/ModelExporter.py#L227)<br><sub>`_validate_source_format(cls, request: ModelExportRequest, *, export_format: str, source: Optional[Path], native_source: bool) -> None`</sub> | method | `ModelExporter` 中校验数据源的内部辅助逻辑。 | `is_file`, `ValueError` |
| [`ModelExporter._validate_path_relationship`](../saddlellm/ModelExporter.py#L262)<br><sub>`_validate_path_relationship(source: Path, destination: Path) -> None`</sub> | method | `ModelExporter` 中校验路径的内部辅助逻辑。 | `source.is_dir`, `ValueError` |
| [`ModelExporter._prepare_destination`](../saddlellm/ModelExporter.py#L269)<br><sub>`_prepare_destination(destination: Path, *, overwrite: bool) -> None`</sub> | method | `ModelExporter` 中准备`prepare_destination`的内部辅助逻辑。 | `Path`, `resolve`, `Path.home`, `ValueError`, `destination.exists`, `any`, `destination.iterdir`, `FileExistsError`, `destination.resolve`, `shutil.rmtree` |
| [`ModelExporter._copy_local_release`](../saddlellm/ModelExporter.py#L284)<br><sub>`_copy_local_release(cls, source: Path, destination: Path) -> None`</sub> | method | `ModelExporter` 中实现发布包的内部辅助逻辑。 | `shutil.copytree` |
| [`ModelExporter._copy_local_release.ignore`](../saddlellm/ModelExporter.py#L285)<br><sub>`ignore(directory: str, names: List[str]) -> List[str]`</sub> | nested function | `ModelExporter` 中实现`ignore`的局部回调/辅助逻辑。 | `ignored.append`, `name.startswith`, `fnmatch.fnmatch` |
| [`ModelExporter._merge_adapter`](../saddlellm/ModelExporter.py#L299)<br><sub>`_merge_adapter(cls, source: Path, destination: Path, request: ModelExportRequest) -> List[str]`</sub> | method | `ModelExporter` 中合并适配器的内部辅助逻辑。 | `stabilize_peft_optional_backends`, `ImportError`, `open`, `json.load`, `adapter_config.get`, `ValueError`, `cls._load_kwargs`, `AutoModelForCausalLM.from_pretrained`, `model.to`, `PeftModel.from_pretrained` |
| [`ModelExporter._materialize_model`](../saddlellm/ModelExporter.py#L348)<br><sub>`_materialize_model(cls, model_path: str, destination: Path, request: ModelExportRequest) -> None`</sub> | method | `ModelExporter` 中实现模型的内部辅助逻辑。 | `ImportError`, `cls._load_kwargs`, `AutoModelForCausalLM.from_pretrained`, `model.to`, `load_tokenizer_compatible`, `model.save_pretrained`, `tokenizer.save_pretrained`, `cls._release_memory` |
| [`ModelExporter._load_kwargs`](../saddlellm/ModelExporter.py#L376)<br><sub>`_load_kwargs(request: ModelExportRequest) -> Tuple[Dict[str, Any], str]`</sub> | method | `ModelExporter` 中加载`load_kwargs`的内部辅助逻辑。 | `torch.cuda.is_available`, `RuntimeError` |
| [`ModelExporter._has_tokenizer_files`](../saddlellm/ModelExporter.py#L403)<br><sub>`_has_tokenizer_files(path: Path) -> bool`</sub> | method | `ModelExporter` 中实现分词器的内部辅助逻辑。 | `any`, `is_file` |
| [`ModelExporter._normalize_tokenizer_config`](../saddlellm/ModelExporter.py#L415)<br><sub>`_normalize_tokenizer_config(destination: Path) -> bool`</sub> | method | `ModelExporter` 中规范化分词器、配置的内部辅助逻辑。 | `config_path.is_file`, `tokenizer_path.is_file`, `json.loads`, `config_path.read_text`, `isinstance`, `config.get`, `open`, `json.dump`, `handle.write` |
| [`ModelExporter._verify_release`](../saddlellm/ModelExporter.py#L433)<br><sub>`_verify_release(cls, destination: Path, *, allow_adapter: bool, format: str='hf') -> None`</sub> | method | `ModelExporter` 中验证发布包的内部辅助逻辑。 | `is_file`, `RuntimeError`, `join`, `cls._has_tokenizer_files`, `any`, `destination.glob`, `path.is_file` |
| [`ModelExporter._write_release_guide`](../saddlellm/ModelExporter.py#L474)<br><sub>`_write_release_guide(destination: Path, adapter_merged: bool, source_type: str, *, format: str='hf') -> None`</sub> | method | `ModelExporter` 中写入发布包的内部辅助逻辑。 | `lower`, `str`, `write_text` |
| [`ModelExporter._inventory`](../saddlellm/ModelExporter.py#L508)<br><sub>`_inventory(cls, destination: Path, *, hash_weights: bool) -> Tuple[List[Dict[str, Any]], int, Dict[str, str]]`</sub> | method | `ModelExporter` 中实现`inventory`的内部辅助逻辑。 | `sorted`, `destination.rglob`, `item.is_file`, `as_posix`, `path.relative_to`, `path.stat`, `inventory.append`, `any`, `fnmatch.fnmatch`, `cls._sha256` |
| [`ModelExporter._sha256`](../saddlellm/ModelExporter.py#L528)<br><sub>`_sha256(path: Path) -> str`</sub> | method | `ModelExporter` 中实现`sha256`的内部辅助逻辑。 | `hashlib.sha256`, `open`, `iter`, `handle.read`, `digest.update`, `digest.hexdigest` |
| [`ModelExporter._release_id`](../saddlellm/ModelExporter.py#L536)<br><sub>`_release_id(inventory: List[Dict[str, Any]], hashes: Dict[str, str]) -> str`</sub> | method | `ModelExporter` 中实现发布包的内部辅助逻辑。 | `encode`, `json.dumps`, `hexdigest`, `hashlib.sha256` |
| [`ModelExporter._release_memory`](../saddlellm/ModelExporter.py#L546)<br><sub>`_release_memory() -> None`</sub> | method | `ModelExporter` 中实现发布包的内部辅助逻辑。 | `gc.collect`, `torch.cuda.is_available`, `torch.cuda.empty_cache` |
| [`export_model`](../saddlellm/ModelExporter.py#L557)<br><sub>`export_model(model_path: str, output_dir: str, **kwargs: Any) -> Dict[str, Any]`</sub> | function | Convenience API returning a JSON-serializable export result. | `to_dict`, `ModelExporter.export`, `ModelExportRequest` |

## `saddlellm/ModelLoader.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`_local_directory`](../saddlellm/ModelLoader.py#L23)<br><sub>`_local_directory(path: PathLike) -> Optional[Path]`</sub> | function | Return an existing local directory without interpreting remote IDs. | `isinstance`, `TypeError`, `expanduser`, `Path`, `candidate.is_dir` |
| [`is_saddle_checkpoint`](../saddlellm/ModelLoader.py#L37)<br><sub>`is_saddle_checkpoint(path: PathLike) -> bool`</sub> | function | Return whether ``path`` is a complete local SaddleLLM checkpoint. | `_local_directory`, `bool`, `is_file` |
| [`_normalize_dtype`](../saddlellm/ModelLoader.py#L53)<br><sub>`_normalize_dtype(dtype: Any) -> Any`</sub> | function | 模块级规范化`normalize_dtype`的内部辅助逻辑。 | `isinstance`, `TypeError`, `replace`, `lower`, `dtype.strip`, `join`, `sorted`, `ValueError` |
| [`_normalize_device`](../saddlellm/ModelLoader.py#L86)<br><sub>`_normalize_device(device: Any, *, allow_auto: bool=True) -> Any`</sub> | function | 模块级规范化设备的内部辅助逻辑。 | `isinstance`, `lower`, `device.strip`, `torch.device`, `torch.cuda.is_available`, `ValueError` |
| [`_native_checkpoint_error`](../saddlellm/ModelLoader.py#L104)<br><sub>`_native_checkpoint_error(path: PathLike) -> Optional[FileNotFoundError]`</sub> | function | Describe a local directory that advertises an incomplete native save. | `_local_directory`, `is_file`, `FileNotFoundError` |
| [`load_causal_lm`](../saddlellm/ModelLoader.py#L118)<br><sub>`load_causal_lm(path: PathLike, *, map_location: Any=None, device: Any=None, dtype: Any=None, trust_remote_code: bool=False, local_files_only: bool=False, **model_kwargs: Any) -> Any`</sub> | function | Load a native SaddleLLM checkpoint or a Hugging Face causal LM. | `_local_directory`, `model_kwargs.pop`, `ValueError`, `_normalize_dtype`, `is_saddle_checkpoint`, `join`, `sorted`, `TypeError`, `SaddleForCausalLM.from_pretrained_saddle`, `str` |
| [`load_model_and_tokenizer`](../saddlellm/ModelLoader.py#L232)<br><sub>`load_model_and_tokenizer(path: PathLike, *, tokenizer_path: Optional[PathLike]=None, map_location: Any=None, device: Any=None, dtype: Any=None, trust_remote_code: bool=False, local_files_only: bool=False, model_kwargs: Optional[Mapping[str, Any]]=None) -> Tuple[Any, Any]`</sub> | function | Load a causal LM and its compatible tokenizer as ``(model, tokenizer)``. | `load_tokenizer_compatible`, `str`, `RuntimeError`, `load_causal_lm`, `dict` |

## `saddlellm/ModelMetrics.py`

共 10 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelMetricSnapshot.to_dict`](../saddlellm/ModelMetrics.py#L16)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ModelMetricSnapshot` 转为可序列化字典。 | `asdict` |
| [`ModelMetricsExtractor.architecture`](../saddlellm/ModelMetrics.py#L24)<br><sub>`architecture(model) -> Dict`</sub> | method | `ModelMetricsExtractor` 中实现`architecture`的公开操作。 | `getattr`, `type`, `hasattr`, `cfg.to_dict`, `dict`, `data.get`, `layer.get`, `get`, `enumerate` |
| [`ModelMetricsExtractor.runtime`](../saddlellm/ModelMetrics.py#L64)<br><sub>`runtime(model) -> Dict`</sub> | method | `ModelMetricsExtractor` 中实现运行时的公开操作。 | `hasattr`, `model.router_metrics`, `ModelMetricsExtractor.cache_profile` |
| [`ModelMetricsExtractor.cache_profile`](../saddlellm/ModelMetrics.py#L72)<br><sub>`cache_profile(model) -> Dict`</sub> | method | `ModelMetricsExtractor` 中实现缓存的公开操作。 | `getattr`, `hasattr`, `cfg.for_layer`, `range`, `len`, `max`, `per_layer.append`, `sum`, `next`, `iter` |
| [`ModelMetricsExtractor.snapshot`](../saddlellm/ModelMetrics.py#L117)<br><sub>`snapshot(model, step: int, logs: Optional[Dict]=None) -> ModelMetricSnapshot`</sub> | method | `ModelMetricsExtractor` 中实现`snapshot`的公开操作。 | `ModelMetricsExtractor.runtime`, `logs.items`, `key.startswith`, `key.endswith`, `ModelMetricSnapshot`, `ModelMetricsExtractor.architecture`, `runtime.get` |
| [`ModelMetricsCallback.__init__`](../saddlellm/ModelMetrics.py#L143)<br><sub>`__init__(self, output_dir: str='./model_metrics', log_every: int=1)`</sub> | method | 初始化 `ModelMetricsCallback` 实例及其运行依赖。 | `max`, `int`, `os.makedirs`, `os.path.join` |
| [`ModelMetricsCallback.on_train_begin`](../saddlellm/ModelMetrics.py#L151)<br><sub>`on_train_begin(self, args, state, control, model=None, **kwargs)`</sub> | method | `ModelMetricsCallback` 中训练`on_train_begin`的公开操作。 | `getattr`, `self._record`, `ModelMetricsExtractor.snapshot`, `int` |
| [`ModelMetricsCallback.on_log`](../saddlellm/ModelMetrics.py#L156)<br><sub>`on_log(self, args, state, control, logs=None, model=None, **kwargs)`</sub> | method | `ModelMetricsCallback` 中记录`on_log`的公开操作。 | `int`, `getattr`, `self._record`, `ModelMetricsExtractor.snapshot` |
| [`ModelMetricsCallback.on_train_end`](../saddlellm/ModelMetrics.py#L167)<br><sub>`on_train_end(self, args, state, control, model=None, **kwargs)`</sub> | method | `ModelMetricsCallback` 中训练`on_train_end`的公开操作。 | `getattr`, `self._record`, `ModelMetricsExtractor.snapshot`, `int` |
| [`ModelMetricsCallback._record`](../saddlellm/ModelMetrics.py#L172)<br><sub>`_record(self, snapshot: ModelMetricSnapshot)`</sub> | method | `ModelMetricsCallback` 中记录`record`的内部辅助逻辑。 | `snapshot.to_dict`, `open`, `f.write`, `json.dumps`, `json.dump` |

## `saddlellm/ModelPruner.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelPruner.__init__`](../saddlellm/ModelPruner.py#L30)<br><sub>`__init__(self, model_name: str='meta-llama/Meta-Llama-3-8B', pruning_method: Literal['l1_unstructured', 'random_unstructured', 'ln_structured', 'head_structured', 'layer_structured']='l1_unstructured', pruning_ratio: float=0.3, global_pruning: bool=True, device_map: str='auto')`</sub> | method | Initialize model pruner. | `AutoModelForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained`, `defaultdict` |
| [`ModelPruner._get_pruning_parameters`](../saddlellm/ModelPruner.py#L73)<br><sub>`_get_pruning_parameters(self) -> List[tuple]`</sub> | method | 获取需要剪枝的参数列表 | `self.model.named_modules`, `isinstance`, `params.append` |
| [`ModelPruner._prune_heads`](../saddlellm/ModelPruner.py#L84)<br><sub>`_prune_heads(self, head_indices: Dict[str, List[int]])`</sub> | method | 剪枝注意力头（结构化） | `head_indices.items`, `self.model.prune_heads`, `int` |
| [`ModelPruner._prune_layers`](../saddlellm/ModelPruner.py#L89)<br><sub>`_prune_layers(self, layer_indices: List[int])`</sub> | method | 剪枝整个Transformer层（结构化） | `hasattr`, `range`, `torch.nn.ModuleList`, `len` |
| [`ModelPruner.fit`](../saddlellm/ModelPruner.py#L99)<br><sub>`fit(self, train_dataset: Optional[Dataset]=None, eval_dataset: Optional[Dataset]=None, epochs: int=1, batch_size: int=2)`</sub> | method | 基于训练数据计算剪枝重要性（可选） :param train_dataset: 用于评估参数重要性的数据 | `train_dataset.map`, `DataCollatorForLanguageModeling`, `self.model.train`, `torch.optim.AdamW`, `self.model.parameters`, `range`, `to`, `self.tokenizer`, `self.model`, `loss.backward` |
| [`ModelPruner.fit.tokenize_fn`](../saddlellm/ModelPruner.py#L114)<br><sub>`tokenize_fn(examples)`</sub> | nested function | `ModelPruner` 中分词`tokenize_fn`的局部回调/辅助逻辑。 | `self.tokenizer` |
| [`ModelPruner.prune`](../saddlellm/ModelPruner.py#L149)<br><sub>`prune(self)`</sub> | method | 执行剪枝操作 | `self._compute_head_importance`, `self._select_heads_to_prune`, `self._prune_heads`, `self._compute_layer_importance`, `self._select_layers_to_prune`, `self._prune_layers`, `self._get_pruning_parameters`, `prune.ln_structured`, `prune.global_unstructured`, `getattr` |
| [`ModelPruner.save`](../saddlellm/ModelPruner.py#L191)<br><sub>`save(self, path: str, remove_masks: bool=False)`</sub> | method | 保存剪枝后的模型 :param remove_masks: 是否永久移除被剪枝的权重（否则只是屏蔽） | `self._get_pruning_parameters`, `prune.remove`, `self.model.save_pretrained`, `self.tokenizer.save_pretrained` |
| [`ModelPruner._compute_head_importance`](../saddlellm/ModelPruner.py#L204)<br><sub>`_compute_head_importance(self) -> Dict[str, float]`</sub> | method | 计算注意力头重要性（示例） | `defaultdict`, `self.model.named_parameters`, `name.split`, `item`, `mean`, `param.abs` |
| [`ModelPruner._compute_layer_importance`](../saddlellm/ModelPruner.py#L214)<br><sub>`_compute_layer_importance(self) -> List[float]`</sub> | method | 计算Transformer层重要性（示例） | — |
| [`ModelPruner.load`](../saddlellm/ModelPruner.py#L219)<br><sub>`load(cls, path: str, **kwargs)`</sub> | method | 加载剪枝后的模型 | `cls` |

## `saddlellm/ModelQuantizer.py`

共 8 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelQuantizer.__init__`](../saddlellm/ModelQuantizer.py#L30)<br><sub>`__init__(self, model_name: str='meta-llama/Meta-Llama-3-8B', method: Literal['dynamic', 'static', 'qat', 'bnb']='dynamic', precision: Literal['int8', 'int4', 'fp4']='int8', device_map: str='auto', use_double_quant: bool=True)`</sub> | method | Initialize quantizer. | `AutoModelForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained`, `self._setup_bnb_config` |
| [`ModelQuantizer._setup_bnb_config`](../saddlellm/ModelQuantizer.py#L67)<br><sub>`_setup_bnb_config(self)`</sub> | method | 配置bitsandbytes量化 | `BitsAndBytesConfig` |
| [`ModelQuantizer.quantize`](../saddlellm/ModelQuantizer.py#L87)<br><sub>`quantize(self, qconfig_spec=None)`</sub> | method | 执行训练后量化（PTQ） :param qconfig_spec: 自定义量化配置（用于static/qat） | `AutoModelForCausalLM.from_pretrained`, `quantize_dynamic`, `self.model.eval`, `torch.quantization.get_default_qconfig`, `torch.quantization.prepare`, `self._calibrate`, `torch.quantization.convert` |
| [`ModelQuantizer.fit`](../saddlellm/ModelQuantizer.py#L128)<br><sub>`fit(self, train_dataset: Dataset, epochs: int=1, batch_size: int=2, learning_rate: float=5e-05)`</sub> | method | 量化感知训练（QAT） | `ValueError`, `self.model.train`, `torch.quantization.get_default_qat_qconfig`, `prepare_qat`, `train_dataset.map`, `DataCollatorForLanguageModeling`, `TrainingArguments`, `torch.cuda.is_bf16_supported`, `Trainer`, `trainer.train` |
| [`ModelQuantizer.fit.tokenize_fn`](../saddlellm/ModelQuantizer.py#L147)<br><sub>`tokenize_fn(examples)`</sub> | nested function | `ModelQuantizer` 中分词`tokenize_fn`的局部回调/辅助逻辑。 | `self.tokenizer` |
| [`ModelQuantizer.save`](../saddlellm/ModelQuantizer.py#L181)<br><sub>`save(self, path: str)`</sub> | method | 保存量化模型 | `self.model.save_pretrained`, `torch.save`, `self.model.state_dict`, `self.tokenizer.save_pretrained` |
| [`ModelQuantizer.load`](../saddlellm/ModelQuantizer.py#L195)<br><sub>`load(cls, path: str, method: str, **kwargs)`</sub> | method | 加载量化模型 | `cls`, `torch.load`, `instance.model.load_state_dict` |
| [`ModelQuantizer._calibrate`](../saddlellm/ModelQuantizer.py#L208)<br><sub>`_calibrate(self, model, calibration_data)`</sub> | method | 静态量化校准（示例） | `model.eval`, `torch.no_grad`, `to`, `self.tokenizer`, `model` |

## `saddlellm/ModelRegistry.py`

共 20 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`_get_torch_dtype`](../saddlellm/ModelRegistry.py#L20)<br><sub>`_get_torch_dtype()`</sub> | function | 模块级读取`get_torch_dtype`的内部辅助逻辑。 | `torch.cuda.is_available`, `torch.cuda.is_bf16_supported` |
| [`ModelSpec.__post_init__`](../saddlellm/ModelRegistry.py#L64)<br><sub>`__post_init__(self)`</sub> | method | 在 `ModelSpec` 创建后校验并规范化字段。 | `self._estimate_params`, `self._chinchilla_optimal_tokens`, `self._compute_flops_per_token`, `self._estimate_active_params` |
| [`ModelSpec._estimate_params`](../saddlellm/ModelRegistry.py#L74)<br><sub>`_estimate_params(self) -> int`</sub> | method | `ModelSpec` 中估算`estimate_params`的内部辅助逻辑。 | — |
| [`ModelSpec._estimate_active_params`](../saddlellm/ModelRegistry.py#L123)<br><sub>`_estimate_active_params(self) -> int`</sub> | method | MoE: 每个 token 实际激活的参数量 | — |
| [`ModelSpec._chinchilla_optimal_tokens`](../saddlellm/ModelRegistry.py#L154)<br><sub>`_chinchilla_optimal_tokens(self) -> int`</sub> | method | `ModelSpec` 中实现`chinchilla_optimal_tokens`的内部辅助逻辑。 | — |
| [`ModelSpec._compute_flops_per_token`](../saddlellm/ModelRegistry.py#L159)<br><sub>`_compute_flops_per_token(self) -> float`</sub> | method | `ModelSpec` 中计算`compute_flops_per_token`的内部辅助逻辑。 | — |
| [`ModelSpec.to_dict`](../saddlellm/ModelRegistry.py#L164)<br><sub>`to_dict(self) -> dict`</sub> | method | 把 `ModelSpec` 转为可序列化字典。 | `d.update` |
| [`ModelSpec.human_params`](../saddlellm/ModelRegistry.py#L182)<br><sub>`human_params(self) -> str`</sub> | method | `ModelSpec` 中实现`human_params`的公开操作。 | — |
| [`ModelSpec.human_tokens`](../saddlellm/ModelRegistry.py#L193)<br><sub>`human_tokens(self) -> str`</sub> | method | `ModelSpec` 中实现`human_tokens`的公开操作。 | — |
| [`ModelRegistry.get`](../saddlellm/ModelRegistry.py#L451)<br><sub>`get(name: str) -> ModelSpec`</sub> | method | 获取模型规格 | `join`, `MODEL_SPECS.keys`, `KeyError` |
| [`ModelRegistry.list_all`](../saddlellm/ModelRegistry.py#L459)<br><sub>`list_all() -> List[ModelSpec]`</sub> | method | 列出所有模型 | `list`, `MODEL_SPECS.values` |
| [`ModelRegistry.list_by_size`](../saddlellm/ModelRegistry.py#L464)<br><sub>`list_by_size(max_params: Optional[int]=None, min_params: Optional[int]=None) -> List[ModelSpec]`</sub> | method | 按参数量筛选 | `MODEL_SPECS.values`, `sorted` |
| [`ModelRegistry.architecture_support`](../saddlellm/ModelRegistry.py#L474)<br><sub>`architecture_support(name_or_spec) -> Dict`</sub> | method | 返回某个模型规格对应的架构支持矩阵。 | `isinstance`, `ModelRegistry.get`, `to_dict`, `ArchitectureRegistry.from_model_spec` |
| [`ModelRegistry.validate_architecture`](../saddlellm/ModelRegistry.py#L482)<br><sub>`validate_architecture(name_or_spec, capability: str='pretrain') -> bool`</sub> | method | 校验某个模型规格是否支持指定能力；不支持时抛出明确错误。 | `isinstance`, `ModelRegistry.get`, `ArchitectureRegistry.require` |
| [`ModelRegistry.create_model`](../saddlellm/ModelRegistry.py#L491)<br><sub>`create_model(spec: ModelSpec, vocab_size_override: Optional[int]=None) -> 'PreTrainedModel'`</sub> | method | 从 ModelSpec 创建 HuggingFace 模型。 支持 Llama, GPT-2, GPT-NeoX 架构。 | `ModelRegistry.validate_architecture`, `LlamaConfig`, `AutoModelForCausalLM.from_config`, `_get_torch_dtype`, `GPT2Config`, `GPTNeoXConfig`, `ImportError`, `Qwen2Config`, `MixtralConfig`, `logger.warning` |
| [`ModelRegistry.create_saddle_model`](../saddlellm/ModelRegistry.py#L628)<br><sub>`create_saddle_model(spec: ModelSpec, vocab_size_override: Optional[int]=None)`</sub> | method | Create SaddleLLM's own modular PyTorch model from a ModelSpec. | `SaddleModelConfig.from_model_spec`, `SaddleForCausalLM` |
| [`ModelRegistry.estimate_memory`](../saddlellm/ModelRegistry.py#L638)<br><sub>`estimate_memory(spec: ModelSpec, batch_size: int=1, seq_length: Optional[int]=None, dtype: str='bf16', optimizer: str='adamw', use_gradient_checkpointing: bool=True) -> Dict[str, float]`</sub> | method | 估算训练/推理所需显存(GB)。 | `round` |
| [`ModelRegistry.compare_specs`](../saddlellm/ModelRegistry.py#L681)<br><sub>`compare_specs(specs: Optional[List[str]]=None) -> str`</sub> | method | 生成模型对比表格 (含架构类型和 MoE 信息) | `list`, `MODEL_SPECS.keys`, `sorted`, `lines.append`, `m.human_params`, `m.human_tokens`, `join` |
| [`ModelRegistry.compare_architectures`](../saddlellm/ModelRegistry.py#L701)<br><sub>`compare_architectures() -> str`</sub> | method | 详细对比架构差异 (Dense vs MoE vs DeepSeek) | — |
| [`ModelRegistry.recommend_for_gpu`](../saddlellm/ModelRegistry.py#L727)<br><sub>`recommend_for_gpu(gpu_memory_gb: int, strategy: str='full_finetune') -> List[ModelSpec]`</sub> | method | 根据 GPU 显存推荐可行的模型 | `sorted`, `MODEL_SPECS.values`, `ModelRegistry.estimate_memory`, `suitable.append` |

## `saddlellm/MultimodalData.py`

共 18 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MultimodalSegment.__post_init__`](../saddlellm/MultimodalData.py#L40)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `MultimodalSegment` 创建后校验并规范化字段。 | `lower`, `str`, `ValueError`, `join`, `sorted` |
| [`MultimodalSegment.to_dict`](../saddlellm/MultimodalData.py#L66)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `MultimodalSegment` 转为可序列化字典。 | `items`, `asdict` |
| [`MultimodalAsset.to_dict`](../saddlellm/MultimodalData.py#L80)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `MultimodalAsset` 转为可序列化字典。 | `asdict` |
| [`MultimodalAsset.to_segment`](../saddlellm/MultimodalData.py#L83)<br><sub>`to_segment(self, *, role: str='input') -> MultimodalSegment`</sub> | method | `MultimodalAsset` 中实现`to_segment`的公开操作。 | `dict`, `MultimodalSegment`, `timing.pop` |
| [`MultimodalSample.to_dict`](../saddlellm/MultimodalData.py#L113)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `MultimodalSample` 转为可序列化字典。 | `image.to_dict`, `segment.to_dict`, `getattr` |
| [`MultimodalSample.validate_episode`](../saddlellm/MultimodalData.py#L134)<br><sub>`validate_episode(self) -> None`</sub> | method | Validate aligned transition fields when episode data is present. | `len`, `ValueError` |
| [`MultimodalNormalizationReport.to_dict`](../saddlellm/MultimodalData.py#L165)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `MultimodalNormalizationReport` 转为可序列化字典。 | `asdict` |
| [`MultimodalDataAdapter.normalize_record`](../saddlellm/MultimodalData.py#L180)<br><sub>`normalize_record(cls, record: Dict, image_root: Optional[str]=None, task: str='mllm_sft') -> Optional[MultimodalSample]`</sub> | method | `MultimodalDataAdapter` 中规范化`normalize_record`的公开操作。 | `cls._extract_segments`, `record.get`, `list`, `MultimodalAsset`, `str`, `cls._extract_messages`, `cls._first`, `strip`, `join`, `messages.append` |
| [`MultimodalDataAdapter.normalize_records`](../saddlellm/MultimodalData.py#L278)<br><sub>`normalize_records(cls, records: Iterable[Dict], image_root: Optional[str]=None, task: str='mllm_sft') -> (List[MultimodalSample], MultimodalNormalizationReport)`</sub> | method | `MultimodalDataAdapter` 中规范化`normalize_records`的公开操作。 | `MultimodalNormalizationReport`, `cls.detect_schema`, `report.detected_schemas.get`, `cls.normalize_record`, `samples.append`, `report.modality_counts.get`, `report.warnings.append` |
| [`MultimodalDataAdapter.normalize_file`](../saddlellm/MultimodalData.py#L305)<br><sub>`normalize_file(cls, input_path: str, output_path: str, image_root: Optional[str]=None, task: str='mllm_sft') -> MultimodalNormalizationReport`</sub> | method | `MultimodalDataAdapter` 中规范化`normalize_file`的公开操作。 | `cls.load_records`, `cls.normalize_records`, `os.path.dirname`, `os.makedirs`, `open`, `f.write`, `json.dumps`, `sample.to_dict`, `json.dump`, `report.to_dict` |
| [`MultimodalDataAdapter.load_records`](../saddlellm/MultimodalData.py#L325)<br><sub>`load_records(path: str) -> List[Dict]`</sub> | method | `MultimodalDataAdapter` 中加载`load_records`的公开操作。 | `lower`, `os.path.splitext`, `open`, `line.strip`, `records.append`, `json.loads`, `json.load`, `isinstance`, `data.get`, `list` |
| [`MultimodalDataAdapter.detect_schema`](../saddlellm/MultimodalData.py#L350)<br><sub>`detect_schema(record: Dict) -> str`</sub> | method | `MultimodalDataAdapter` 中检测`detect_schema`的公开操作。 | `any` |
| [`MultimodalDataAdapter._extract_messages`](../saddlellm/MultimodalData.py#L366)<br><sub>`_extract_messages(cls, record: Dict) -> List[Dict[str, str]]`</sub> | method | `MultimodalDataAdapter` 中提取`extract_messages`的内部辅助逻辑。 | `record.get`, `lower`, `str`, `item.get`, `strip`, `messages.append` |
| [`MultimodalDataAdapter._extract_images`](../saddlellm/MultimodalData.py#L383)<br><sub>`_extract_images(cls, record: Dict, image_root: Optional[str]) -> List[MultimodalAsset]`</sub> | method | `MultimodalDataAdapter` 中提取`extract_images`的内部辅助逻辑。 | `MultimodalAsset`, `str`, `cls._extract_segments` |
| [`MultimodalDataAdapter._extract_segments`](../saddlellm/MultimodalData.py#L391)<br><sub>`_extract_segments(cls, record: Dict, media_root: Optional[str]) -> List[MultimodalSegment]`</sub> | method | `MultimodalDataAdapter` 中提取`extract_segments`的内部辅助逻辑。 | `record.get`, `isinstance`, `ValueError`, `lower`, `str`, `item.get`, `cls._resolve_media_path`, `dict`, `metadata.update`, `item.items` |
| [`MultimodalDataAdapter._resolve_media_path`](../saddlellm/MultimodalData.py#L471)<br><sub>`_resolve_media_path(path: str, media_root: Optional[str]) -> str`</sub> | method | `MultimodalDataAdapter` 中解析路径的内部辅助逻辑。 | `os.path.isabs`, `path.startswith`, `os.path.abspath`, `os.path.join` |
| [`MultimodalDataAdapter._first`](../saddlellm/MultimodalData.py#L479)<br><sub>`_first(record: Dict, keys: Sequence[str]) -> str`</sub> | method | `MultimodalDataAdapter` 中实现`first`的内部辅助逻辑。 | `record.get`, `isinstance`, `str` |
| [`normalize_multimodal_file`](../saddlellm/MultimodalData.py#L489)<br><sub>`normalize_multimodal_file(input_path: str, output_path: str, image_root: Optional[str]=None, task: str='mllm_sft') -> Dict`</sub> | function | 模块级规范化`normalize_multimodal_file`的公开操作。 | `to_dict`, `MultimodalDataAdapter.normalize_file` |

## `saddlellm/MultimodalModeling.py`

共 13 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MultimodalConfig.__post_init__`](../saddlellm/MultimodalModeling.py#L26)<br><sub>`__post_init__(self)`</sub> | method | 在 `MultimodalConfig` 创建后校验并规范化字段。 | `MultimodalProjectorConfig` |
| [`MultimodalConfig.to_dict`](../saddlellm/MultimodalModeling.py#L33)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `MultimodalConfig` 转为可序列化字典。 | `self.llm_config.to_dict`, `self.vision_config.to_dict`, `self.projector_config.to_dict` |
| [`MultimodalConfig.from_dict`](../saddlellm/MultimodalModeling.py#L46)<br><sub>`from_dict(cls, raw: Dict) -> 'MultimodalConfig'`</sub> | method | 从字典解析并创建 `MultimodalConfig`。 | `cls`, `SaddleModelConfig`, `raw.get`, `VisionBackboneConfig`, `MultimodalProjectorConfig` |
| [`MultimodalForCausalLM.__init__`](../saddlellm/MultimodalModeling.py#L62)<br><sub>`__init__(self, config: MultimodalConfig)`</sub> | method | 初始化 `MultimodalForCausalLM` 实例及其运行依赖。 | `__init__`, `super`, `SaddleForCausalLM`, `VisionBackboneRegistry.build`, `getattr`, `MultimodalProjectorConfig`, `MultimodalProjectorFactory.build`, `self._apply_freezing` |
| [`MultimodalForCausalLM.from_text_blueprint`](../saddlellm/MultimodalModeling.py#L76)<br><sub>`from_text_blueprint(cls, blueprint, vision_config: Optional[VisionBackboneConfig]=None, projector_config: Optional[MultimodalProjectorConfig]=None, image_token_id: Optional[int]=None) -> 'MultimodalForCausalLM'`</sub> | method | `MultimodalForCausalLM` 中实现模型蓝图的公开操作。 | `SaddleModelConfig.from_blueprint`, `cls`, `MultimodalConfig`, `VisionBackboneConfig` |
| [`MultimodalForCausalLM._apply_freezing`](../saddlellm/MultimodalModeling.py#L94)<br><sub>`_apply_freezing(self)`</sub> | method | `MultimodalForCausalLM` 中应用`apply_freezing`的内部辅助逻辑。 | `self.vision_encoder.parameters`, `param.requires_grad_`, `self.language_model.parameters`, `self.projector.parameters` |
| [`MultimodalForCausalLM.encode_images`](../saddlellm/MultimodalModeling.py#L104)<br><sub>`encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor`</sub> | method | `MultimodalForCausalLM` 中编码`encode_images`的公开操作。 | `self.vision_encoder`, `self.projector` |
| [`MultimodalForCausalLM.prepare_multimodal_inputs`](../saddlellm/MultimodalModeling.py#L108)<br><sub>`prepare_multimodal_inputs(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]`</sub> | method | `MultimodalForCausalLM` 中准备`prepare_multimodal_inputs`的公开操作。 | `self.language_model.get_input_embeddings`, `to`, `self.encode_images`, `self._prepend_image_tokens`, `self._replace_image_tokens` |
| [`MultimodalForCausalLM.forward`](../saddlellm/MultimodalModeling.py#L124)<br><sub>`forward(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, return_dict: bool=True, **kwargs) -> SaddleCausalLMOutput`</sub> | method | 执行 `MultimodalForCausalLM` 的前向计算。 | `self.prepare_multimodal_inputs`, `self.language_model`, `prepared.get` |
| [`MultimodalForCausalLM._prepend_image_tokens`](../saddlellm/MultimodalModeling.py#L148)<br><sub>`_prepend_image_tokens(self, text_embeds, image_embeds, attention_mask, labels)`</sub> | method | `MultimodalForCausalLM` 中实现图像的内部辅助逻辑。 | `torch.cat`, `torch.ones`, `torch.full` |
| [`MultimodalForCausalLM._replace_image_tokens`](../saddlellm/MultimodalModeling.py#L160)<br><sub>`_replace_image_tokens(self, input_ids, text_embeds, image_embeds, attention_mask, labels)`</sub> | method | `MultimodalForCausalLM` 中实现图像的内部辅助逻辑。 | `text_embeds.clone`, `labels.clone`, `range`, `flatten`, `nonzero`, `positions.numel`, `min`, `int`, `to`, `torch.ones` |
| [`MultimodalForCausalLM.save_pretrained`](../saddlellm/MultimodalModeling.py#L176)<br><sub>`save_pretrained(self, path: str, state_dict: Optional[Dict]=None, **_) -> str`</sub> | method | 保存 `MultimodalForCausalLM`，遵循预训练模型的目录契约。 | `os.makedirs`, `open`, `os.path.join`, `json.dump`, `self.config.to_dict`, `torch.save`, `self.state_dict` |
| [`MultimodalForCausalLM.from_pretrained_saddle_multimodal`](../saddlellm/MultimodalModeling.py#L184)<br><sub>`from_pretrained_saddle_multimodal(cls, path: str, map_location: Optional[str]=None) -> 'MultimodalForCausalLM'`</sub> | method | `MultimodalForCausalLM` 中实现`from_pretrained_saddle_multimodal`的公开操作。 | `os.path.join`, `os.path.exists`, `open`, `MultimodalConfig.from_dict`, `json.load`, `cls`, `torch.load`, `model.load_state_dict` |

## `saddlellm/MultimodalProjector.py`

共 10 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MultimodalProjectorConfig.to_dict`](../saddlellm/MultimodalProjector.py#L19)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `MultimodalProjectorConfig` 转为可序列化字典。 | `asdict` |
| [`LinearProjector.__init__`](../saddlellm/MultimodalProjector.py#L24)<br><sub>`__init__(self, config: MultimodalProjectorConfig)`</sub> | method | 初始化 `LinearProjector` 实例及其运行依赖。 | `__init__`, `super`, `nn.Linear` |
| [`LinearProjector.forward`](../saddlellm/MultimodalProjector.py#L28)<br><sub>`forward(self, features: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `LinearProjector` 的前向计算。 | `self.proj` |
| [`MLPProjector.__init__`](../saddlellm/MultimodalProjector.py#L33)<br><sub>`__init__(self, config: MultimodalProjectorConfig)`</sub> | method | 初始化 `MLPProjector` 实例及其运行依赖。 | `__init__`, `super`, `max`, `nn.Sequential`, `nn.Linear`, `nn.GELU`, `nn.Dropout` |
| [`MLPProjector.forward`](../saddlellm/MultimodalProjector.py#L43)<br><sub>`forward(self, features: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `MLPProjector` 的前向计算。 | `self.net` |
| [`GatedMLPProjector.__init__`](../saddlellm/MultimodalProjector.py#L48)<br><sub>`__init__(self, config: MultimodalProjectorConfig)`</sub> | method | 初始化 `GatedMLPProjector` 实例及其运行依赖。 | `__init__`, `super`, `max`, `nn.Linear`, `nn.Dropout` |
| [`GatedMLPProjector.forward`](../saddlellm/MultimodalProjector.py#L56)<br><sub>`forward(self, features: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `GatedMLPProjector` 的前向计算。 | `self.down`, `self.dropout`, `F.silu`, `self.gate`, `self.up` |
| [`ResamplerProjector.__init__`](../saddlellm/MultimodalProjector.py#L63)<br><sub>`__init__(self, config: MultimodalProjectorConfig)`</sub> | method | 初始化 `ResamplerProjector` 实例及其运行依赖。 | `__init__`, `super`, `nn.Parameter`, `torch.randn`, `nn.Linear` |
| [`ResamplerProjector.forward`](../saddlellm/MultimodalProjector.py#L70)<br><sub>`forward(self, features: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `ResamplerProjector` 的前向计算。 | `expand`, `self.k_proj`, `self.v_proj`, `torch.matmul`, `k.transpose`, `to`, `F.softmax`, `scores.float`, `self.out` |
| [`MultimodalProjectorFactory.build`](../saddlellm/MultimodalProjector.py#L82)<br><sub>`build(config: MultimodalProjectorConfig) -> nn.Module`</sub> | method | `MultimodalProjectorFactory` 中构建`build`的公开操作。 | `lower`, `LinearProjector`, `MLPProjector`, `GatedMLPProjector`, `ResamplerProjector`, `ValueError` |

## `saddlellm/MusicCodeModel.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MusicCodeConfig.__post_init__`](../saddlellm/MusicCodeModel.py#L25)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `MusicCodeConfig` 创建后校验并规范化字段。 | `int`, `getattr`, `ValueError` |
| [`MusicCodeConfig.to_dict`](../saddlellm/MusicCodeModel.py#L42)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `MusicCodeConfig` 转为可序列化字典。 | `asdict` |
| [`ConditionalMusicCodeTransformer.__init__`](../saddlellm/MusicCodeModel.py#L49)<br><sub>`__init__(self, config: MusicCodeConfig) -> None`</sub> | method | 初始化 `ConditionalMusicCodeTransformer` 实例及其运行依赖。 | `__init__`, `super`, `nn.ModuleList`, `nn.Embedding`, `range`, `nn.Sequential`, `nn.LayerNorm`, `nn.Linear`, `nn.TransformerEncoderLayer`, `int` |
| [`ConditionalMusicCodeTransformer.forward`](../saddlellm/MusicCodeModel.py#L83)<br><sub>`forward(self, audio_codes: torch.Tensor, condition: torch.Tensor, *, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor`</sub> | method | 执行 `ConditionalMusicCodeTransformer` 的前向计算。 | `self._validate_inputs`, `sum`, `embedding`, `enumerate`, `torch.arange`, `self.position_embedding`, `to`, `self.condition_projection`, `torch.triu`, `torch.ones` |
| [`ConditionalMusicCodeTransformer.compute_loss`](../saddlellm/MusicCodeModel.py#L121)<br><sub>`compute_loss(self, audio_codes: torch.Tensor, condition: torch.Tensor, *, attention_mask: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]`</sub> | method | `ConditionalMusicCodeTransformer` 中计算`compute_loss`的公开操作。 | `self._validate_inputs`, `ValueError`, `torch.full`, `torch.cat`, `torch.ones_like`, `self`, `range`, `reshape`, `F.cross_entropy`, `clamp_min` |
| [`ConditionalMusicCodeTransformer.generate`](../saddlellm/MusicCodeModel.py#L174)<br><sub>`generate(self, condition: torch.Tensor, *, max_frames: int, prompt_codes: Optional[torch.Tensor]=None, temperature: float=1.0, top_k: Optional[int]=None) -> torch.Tensor`</sub> | method | `ConditionalMusicCodeTransformer` 中生成`generate`的公开操作。 | `ValueError`, `torch.full`, `self._validate_inputs`, `torch.cat`, `prompt_codes.clone`, `self.eval`, `self`, `logits.argmax`, `torch.topk`, `reshape` |
| [`ConditionalMusicCodeTransformer._validate_inputs`](../saddlellm/MusicCodeModel.py#L228)<br><sub>`_validate_inputs(self, audio_codes: torch.Tensor, condition: torch.Tensor, *, attention_mask: Optional[torch.Tensor]=None) -> None`</sub> | method | `ConditionalMusicCodeTransformer` 中校验`validate_inputs`的内部辅助逻辑。 | `ValueError`, `audio_codes.numel`, `audio_codes.min`, `audio_codes.max` |

## `saddlellm/MusicCodeTrainer.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MusicCodeTrainingConfig.__post_init__`](../saddlellm/MusicCodeTrainer.py#L39)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `MusicCodeTrainingConfig` 创建后校验并规范化字段。 | `int`, `getattr`, `ValueError` |
| [`CachedMusicCodeDataset.__init__`](../saddlellm/MusicCodeTrainer.py#L56)<br><sub>`__init__(self, path: str) -> None`</sub> | method | 初始化 `CachedMusicCodeDataset` 实例及其运行依赖。 | `Path`, `source.is_dir`, `source.suffix.lower`, `ShardedNpzStore`, `self._sharded.manifest.get`, `self._sharded.array_shape`, `int`, `self._sharded.array_stat`, `ValueError`, `source.is_file` |
| [`CachedMusicCodeDataset.__len__`](../saddlellm/MusicCodeTrainer.py#L116)<br><sub>`__len__(self) -> int`</sub> | method | `CachedMusicCodeDataset` 中实现`len__`的内部辅助逻辑。 | `len` |
| [`CachedMusicCodeDataset.__getitem__`](../saddlellm/MusicCodeTrainer.py#L119)<br><sub>`__getitem__(self, index: int) -> Dict[str, torch.Tensor]`</sub> | method | `CachedMusicCodeDataset` 中实现`getitem__`的内部辅助逻辑。 | `self._sharded.get`, `long`, `torch.from_numpy`, `float`, `np.array` |
| [`CachedMusicCodeDataset.infer_model_config`](../saddlellm/MusicCodeTrainer.py#L143)<br><sub>`infer_model_config(self, **overrides: Any) -> MusicCodeConfig`</sub> | method | `CachedMusicCodeDataset` 中推断模型、配置的公开操作。 | `MusicCodeConfig`, `int` |
| [`_resolve_device`](../saddlellm/MusicCodeTrainer.py#L154)<br><sub>`_resolve_device(value: str) -> torch.device`</sub> | function | 模块级解析设备的内部辅助逻辑。 | `torch.device`, `torch.cuda.is_available`, `RuntimeError` |
| [`_autocast`](../saddlellm/MusicCodeTrainer.py#L163)<br><sub>`_autocast(device: torch.device, precision: str)`</sub> | function | 模块级实现`autocast`的内部辅助逻辑。 | `torch.autocast` |
| [`_lr_multiplier`](../saddlellm/MusicCodeTrainer.py#L170)<br><sub>`_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float`</sub> | function | 模块级实现`lr_multiplier`的内部辅助逻辑。 | `max`, `math.cos`, `min` |
| [`save_music_code_checkpoint`](../saddlellm/MusicCodeTrainer.py#L177)<br><sub>`save_music_code_checkpoint(model: ConditionalMusicCodeTransformer, path: str, *, training: Optional[MusicCodeTrainingConfig]=None, global_step: int=0, optimizer: Optional[torch.optim.Optimizer]=None, scheduler: Optional[torch.optim.lr_scheduler.LambdaLR]=None) -> str`</sub> | function | 模块级保存检查点的公开操作。 | `os.makedirs`, `torch.save`, `model.state_dict`, `os.path.join`, `model.config.to_dict`, `int`, `asdict`, `open`, `json.dump`, `file.write` |
| [`load_music_code_checkpoint`](../saddlellm/MusicCodeTrainer.py#L210)<br><sub>`load_music_code_checkpoint(path: str, *, map_location: str='cpu') -> ConditionalMusicCodeTransformer`</sub> | function | 模块级加载检查点的公开操作。 | `Path`, `config_path.is_file`, `weights_path.is_file`, `FileNotFoundError`, `json.loads`, `config_path.read_text`, `ConditionalMusicCodeTransformer`, `MusicCodeConfig`, `model.load_state_dict`, `torch.load` |
| [`train_music_code`](../saddlellm/MusicCodeTrainer.py#L230)<br><sub>`train_music_code(model_config: MusicCodeConfig, training_config: MusicCodeTrainingConfig) -> Dict[str, Any]`</sub> | function | 模块级训练`train_music_code`的公开操作。 | `random.seed`, `np.random.seed`, `torch.manual_seed`, `CachedMusicCodeDataset`, `ValueError`, `_resolve_device`, `to`, `ConditionalMusicCodeTransformer`, `DataLoader`, `math.ceil` |

## `saddlellm/NativePostTraining.py`

共 21 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`_device`](../saddlellm/NativePostTraining.py#L26)<br><sub>`_device() -> str`</sub> | function | 模块级实现设备的内部辅助逻辑。 | `torch.cuda.is_available` |
| [`_format_messages`](../saddlellm/NativePostTraining.py#L30)<br><sub>`_format_messages(tokenizer: Any, messages: Sequence[Dict[str, str]], *, add_generation_prompt: bool) -> str`</sub> | function | 模块级格式化`format_messages`的内部辅助逻辑。 | `hasattr`, `tokenizer.apply_chat_template`, `list`, `message.get`, `lines.append`, `join` |
| [`_encode`](../saddlellm/NativePostTraining.py#L46)<br><sub>`_encode(tokenizer: Any, text: str, max_length: int) -> List[int]`</sub> | function | 模块级编码`encode`的内部辅助逻辑。 | `tokenizer`, `list` |
| [`_common_prefix_length`](../saddlellm/NativePostTraining.py#L56)<br><sub>`_common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int`</sub> | function | 模块级实现`common_prefix_length`的内部辅助逻辑。 | `zip` |
| [`_sft_features`](../saddlellm/NativePostTraining.py#L65)<br><sub>`_sft_features(record: Dict[str, Any], tokenizer: Any, max_length: int) -> Optional[Dict[str, List[int]]]`</sub> | function | 模块级实现`sft_features`的内部辅助逻辑。 | `PostTrainingDataAdapter.normalize_record`, `normalized.get`, `_encode`, `str`, `len`, `list`, `enumerate`, `message.get`, `_format_messages`, `_common_prefix_length` |
| [`_preference_features`](../saddlellm/NativePostTraining.py#L97)<br><sub>`_preference_features(record: Dict[str, Any], tokenizer: Any, max_length: int, max_prompt_length: int) -> Optional[Dict[str, List[int]]]`</sub> | function | 模块级实现`preference_features`的内部辅助逻辑。 | `PostTrainingDataAdapter.normalize_record`, `str`, `completion`, `any` |
| [`_preference_features.completion`](../saddlellm/NativePostTraining.py#L105)<br><sub>`completion(value: Any) -> Dict[str, List[int]]`</sub> | nested function | 模块级实现`completion`的局部回调/辅助逻辑。 | `str`, `prompt_text.endswith`, `_encode`, `min`, `_common_prefix_length`, `len`, `max` |
| [`_FeatureDataset.__init__`](../saddlellm/NativePostTraining.py#L132)<br><sub>`__init__(self, features: Sequence[Dict[str, List[int]]]) -> None`</sub> | method | 初始化 `_FeatureDataset` 实例及其运行依赖。 | `list` |
| [`_FeatureDataset.__len__`](../saddlellm/NativePostTraining.py#L135)<br><sub>`__len__(self) -> int`</sub> | method | `_FeatureDataset` 中实现`len__`的内部辅助逻辑。 | `len` |
| [`_FeatureDataset.__getitem__`](../saddlellm/NativePostTraining.py#L138)<br><sub>`__getitem__(self, index: int) -> Dict[str, List[int]]`</sub> | method | `_FeatureDataset` 中实现`getitem__`的内部辅助逻辑。 | — |
| [`NativeSFTCollator.__init__`](../saddlellm/NativePostTraining.py#L143)<br><sub>`__init__(self, tokenizer: Any) -> None`</sub> | method | 初始化 `NativeSFTCollator` 实例及其运行依赖。 | `ValueError` |
| [`NativeSFTCollator.__call__`](../saddlellm/NativePostTraining.py#L151)<br><sub>`__call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]`</sub> | method | `NativeSFTCollator` 中实现`call__`的内部辅助逻辑。 | `max`, `len`, `input_ids.append`, `attention_mask.append`, `labels.append`, `torch.tensor` |
| [`NativeDPOCollator.__init__`](../saddlellm/NativePostTraining.py#L167)<br><sub>`__init__(self, tokenizer: Any) -> None`</sub> | method | 初始化 `NativeDPOCollator` 实例及其运行依赖。 | `ValueError` |
| [`NativeDPOCollator.__call__`](../saddlellm/NativePostTraining.py#L175)<br><sub>`__call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]`</sub> | method | `NativeDPOCollator` 中实现`call__`的内部辅助逻辑。 | `max`, `len`, `ids.append`, `masks.append`, `labels.append`, `torch.tensor` |
| [`_sequence_log_probs`](../saddlellm/NativePostTraining.py#L192)<br><sub>`_sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor`</sub> | function | 模块级记录`sequence_log_probs`的内部辅助逻辑。 | `float`, `shifted_labels.ne`, `shifted_labels.masked_fill`, `squeeze`, `gather`, `shifted_logits.log_softmax`, `safe_labels.unsqueeze`, `sum` |
| [`NativeDPOTrainer.__init__`](../saddlellm/NativePostTraining.py#L204)<br><sub>`__init__(self, *args: Any, reference_model: torch.nn.Module, beta: float, **kwargs: Any) -> None`</sub> | method | 初始化 `NativeDPOTrainer` 实例及其运行依赖。 | `__init__`, `super`, `reference_model.eval`, `self.reference_model.requires_grad_`, `float` |
| [`NativeDPOTrainer.compute_loss`](../saddlellm/NativePostTraining.py#L210)<br><sub>`compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)`</sub> | method | `NativeDPOTrainer` 中计算`compute_loss`的公开操作。 | `model`, `_sequence_log_probs`, `next`, `self.reference_model.parameters`, `self.reference_model.to`, `torch.no_grad`, `self.reference_model`, `mean`, `F.logsigmoid`, `detach` |
| [`_training_arguments`](../saddlellm/NativePostTraining.py#L293)<br><sub>`_training_arguments(config: Any, *, remove_unused_columns: bool) -> TrainingArguments`</sub> | function | 模块级实现训练的内部辅助逻辑。 | `_device`, `ValueError`, `bool`, `TrainingArguments`, `max`, `torch.cuda.is_bf16_supported` |
| [`_split_features`](../saddlellm/NativePostTraining.py#L329)<br><sub>`_split_features(features: List[Dict[str, List[int]]], fraction: float, seed: int)`</sub> | function | 模块级切分`split_features`的内部辅助逻辑。 | `_FeatureDataset`, `len`, `ValueError`, `manual_seed`, `torch.Generator`, `tolist`, `torch.randperm`, `max`, `int`, `math.ceil` |
| [`train_native_sft`](../saddlellm/NativePostTraining.py#L343)<br><sub>`train_native_sft(config: NativeSFTConfig) -> Dict[str, Any]`</sub> | function | 模块级训练`train_native_sft`的公开操作。 | `SaddleModelAdapter`, `adapter.validate_stage`, `adapter.load_tokenizer`, `PostTrainingDataAdapter.load_records`, `_sft_features`, `ValueError`, `_split_features`, `adapter.load_model`, `SaddleTrainer`, `_training_arguments` |
| [`train_native_dpo`](../saddlellm/NativePostTraining.py#L384)<br><sub>`train_native_dpo(config: NativeDPOConfig) -> Dict[str, Any]`</sub> | function | 模块级训练`train_native_dpo`的公开操作。 | `SaddleModelAdapter`, `adapter.validate_stage`, `adapter.load_tokenizer`, `PostTrainingDataAdapter.load_records`, `_preference_features`, `ValueError`, `_split_features`, `adapter.load_model`, `copy.deepcopy`, `NativeDPOTrainer` |

## `saddlellm/NativeTrainer.py`

共 1 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SaddleTrainer._save`](../saddlellm/NativeTrainer.py#L20)<br><sub>`_save(self, output_dir: Optional[str]=None, state_dict: Optional[Dict]=None) -> None`</sub> | method | `SaddleTrainer` 中保存`save`的内部辅助逻辑。 | `os.makedirs`, `self.accelerator.unwrap_model`, `hasattr`, `TypeError`, `model.state_dict`, `current_distributed_runtime`, `getattr`, `model.save_pretrained`, `int`, `processing_class.save_pretrained` |

## `saddlellm/OnPolicyDistillation.py`

共 10 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TeacherPolicySignal.to_dict`](../saddlellm/OnPolicyDistillation.py#L40)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `TeacherPolicySignal` 转为可序列化字典。 | `asdict` |
| [`MultiTeacherOnPolicyDistiller.__init__`](../saddlellm/OnPolicyDistillation.py#L66)<br><sub>`__init__(self, teachers: Sequence[OnPolicyTeacherSpec], student_model=None, student_tokenizer=None, rollout_fn: Optional[StudentRolloutFn]=None, score_fn: Optional[TeacherScoreFn]=None, config: Optional[MOPDConfig]=None)`</sub> | method | 初始化 `MultiTeacherOnPolicyDistiller` 实例及其运行依赖。 | `ValueError`, `list`, `MOPDConfig`, `random.seed` |
| [`MultiTeacherOnPolicyDistiller.collect`](../saddlellm/OnPolicyDistillation.py#L86)<br><sub>`collect(self, prompts: Sequence[str]) -> List[Dict]`</sub> | method | `MultiTeacherOnPolicyDistiller` 中收集`collect`的公开操作。 | `expanded.extend`, `self._student_rollout`, `zip`, `self._teacher_signal`, `self._aggregate`, `signal.to_dict`, `len`, `batch_records.append`, `self.records.extend` |
| [`MultiTeacherOnPolicyDistiller.save_jsonl`](../saddlellm/OnPolicyDistillation.py#L119)<br><sub>`save_jsonl(self, path: Optional[str]=None, records: Optional[Iterable[Dict]]=None) -> str`</sub> | method | `MultiTeacherOnPolicyDistiller` 中保存`save_jsonl`的公开操作。 | `os.path.join`, `os.makedirs`, `os.path.dirname`, `list`, `open`, `f.write`, `json.dumps` |
| [`MultiTeacherOnPolicyDistiller.planning_summary`](../saddlellm/OnPolicyDistillation.py#L128)<br><sub>`planning_summary(self) -> Dict`</sub> | method | `MultiTeacherOnPolicyDistiller` 中实现`planning_summary`的公开操作。 | `len`, `asdict` |
| [`MultiTeacherOnPolicyDistiller._student_rollout`](../saddlellm/OnPolicyDistillation.py#L145)<br><sub>`_student_rollout(self, prompts: List[str]) -> List[str]`</sub> | method | `MultiTeacherOnPolicyDistiller` 中实现`student_rollout`的内部辅助逻辑。 | `self.rollout_fn`, `next`, `self.student_model.parameters`, `self.student_model.eval`, `to`, `self.student_tokenizer`, `torch.no_grad`, `self.student_model.generate`, `self.student_tokenizer.decode`, `responses.append` |
| [`MultiTeacherOnPolicyDistiller._teacher_signal`](../saddlellm/OnPolicyDistillation.py#L177)<br><sub>`_teacher_signal(self, spec: OnPolicyTeacherSpec, prompt: str, student_response: str) -> TeacherPolicySignal`</sub> | method | `MultiTeacherOnPolicyDistiller` 中实现`teacher_signal`的内部辅助逻辑。 | `self._build_teacher_prompt`, `spec.teacher.generate`, `self._score`, `TeacherPolicySignal` |
| [`MultiTeacherOnPolicyDistiller._build_teacher_prompt`](../saddlellm/OnPolicyDistillation.py#L203)<br><sub>`_build_teacher_prompt(self, spec: OnPolicyTeacherSpec, prompt: str, student_response: str) -> str`</sub> | method | `MultiTeacherOnPolicyDistiller` 中构建提示词的内部辅助逻辑。 | — |
| [`MultiTeacherOnPolicyDistiller._score`](../saddlellm/OnPolicyDistillation.py#L211)<br><sub>`_score(self, prompt: str, student_response: str, teacher_response: str) -> float`</sub> | method | `MultiTeacherOnPolicyDistiller` 中评分`score`的内部辅助逻辑。 | `float`, `self.score_fn`, `min`, `len` |
| [`MultiTeacherOnPolicyDistiller._aggregate`](../saddlellm/OnPolicyDistillation.py#L220)<br><sub>`_aggregate(self, signals: Sequence[TeacherPolicySignal]) -> TeacherPolicySignal`</sub> | method | `MultiTeacherOnPolicyDistiller` 中实现`aggregate`的内部辅助逻辑。 | `sum`, `max`, `random.choice`, `list`, `random.random` |

## `saddlellm/OperatorBackends.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`AttentionBackendSpec.to_dict`](../saddlellm/OperatorBackends.py#L30)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `AttentionBackendSpec` 转为可序列化字典。 | `asdict` |
| [`AttentionBackendRegistry.register`](../saddlellm/OperatorBackends.py#L41)<br><sub>`register(cls, spec: AttentionBackendSpec, handler: AttentionFn) -> None`</sub> | method | `AttentionBackendRegistry` 中注册`register`的公开操作。 | — |
| [`AttentionBackendRegistry.list_backends`](../saddlellm/OperatorBackends.py#L46)<br><sub>`list_backends(cls) -> Dict[str, Dict]`</sub> | method | `AttentionBackendRegistry` 中列出`list_backends`的公开操作。 | `cls.ensure_defaults`, `spec.to_dict`, `cls._registry.items` |
| [`AttentionBackendRegistry.available_backends`](../saddlellm/OperatorBackends.py#L51)<br><sub>`available_backends(cls) -> List[str]`</sub> | method | `AttentionBackendRegistry` 中实现`available_backends`的公开操作。 | `cls.ensure_defaults`, `cls._registry.items` |
| [`AttentionBackendRegistry.resolve`](../saddlellm/OperatorBackends.py#L56)<br><sub>`resolve(cls, name: str) -> AttentionBackendSpec`</sub> | method | `AttentionBackendRegistry` 中解析`resolve`的公开操作。 | `cls.ensure_defaults`, `lower`, `cls._registry.get`, `ValueError`, `cls.resolve` |
| [`AttentionBackendRegistry.run`](../saddlellm/OperatorBackends.py#L76)<br><sub>`run(cls, name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor`</sub> | method | `AttentionBackendRegistry` 中执行`run`的公开操作。 | `cls.resolve`, `cls._handlers.get`, `ValueError`, `handler` |
| [`AttentionBackendRegistry.ensure_defaults`](../saddlellm/OperatorBackends.py#L92)<br><sub>`ensure_defaults(cls) -> None`</sub> | method | `AttentionBackendRegistry` 中确保`ensure_defaults`的公开操作。 | `hasattr`, `cls.register`, `AttentionBackendSpec`, `importlib.util.find_spec` |
| [`_sdpa_attention`](../saddlellm/OperatorBackends.py#L158)<br><sub>`_sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor`</sub> | function | 模块级实现注意力的内部辅助逻辑。 | `F.scaled_dot_product_attention` |
| [`_eager_attention`](../saddlellm/OperatorBackends.py#L168)<br><sub>`_eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor`</sub> | function | 模块级实现注意力的内部辅助逻辑。 | `torch.matmul`, `k.transpose`, `math.sqrt`, `torch.triu`, `torch.full`, `torch.finfo`, `to`, `F.softmax`, `scores.float` |
| [`_xformers_attention`](../saddlellm/OperatorBackends.py#L189)<br><sub>`_xformers_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor`</sub> | function | 模块级实现注意力的内部辅助逻辑。 | `_sdpa_attention`, `xops.LowerTriangularMask`, `xops.memory_efficient_attention`, `q.transpose`, `k.transpose`, `v.transpose`, `out.transpose` |
| [`list_attention_backends`](../saddlellm/OperatorBackends.py#L213)<br><sub>`list_attention_backends() -> Dict[str, Dict]`</sub> | function | Return registered attention backend capabilities. | `AttentionBackendRegistry.list_backends` |

## `saddlellm/PeftSFTTrainer.py`

共 14 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`_device`](../saddlellm/PeftSFTTrainer.py#L75)<br><sub>`_device() -> str`</sub> | function | 模块级实现设备的内部辅助逻辑。 | `torch.cuda.is_available`, `hasattr`, `torch.backends.mps.is_available` |
| [`_supports_bf16`](../saddlellm/PeftSFTTrainer.py#L83)<br><sub>`_supports_bf16() -> bool`</sub> | function | 模块级实现`supports_bf16`的内部辅助逻辑。 | `torch.cuda.is_available`, `torch.cuda.is_bf16_supported` |
| [`_dtype_for_device`](../saddlellm/PeftSFTTrainer.py#L87)<br><sub>`_dtype_for_device(device: str)`</sub> | function | 模块级实现设备的内部辅助逻辑。 | `_supports_bf16` |
| [`_load_sft_dataset`](../saddlellm/PeftSFTTrainer.py#L93)<br><sub>`_load_sft_dataset(dataset_path: str)`</sub> | function | 模块级加载数据集的内部辅助逻辑。 | `os.path.isdir`, `load_dataset`, `lower`, `os.path.splitext` |
| [`_format_messages`](../saddlellm/PeftSFTTrainer.py#L106)<br><sub>`_format_messages(tokenizer, messages) -> str`</sub> | function | 模块级格式化`format_messages`的内部辅助逻辑。 | `hasattr`, `tokenizer.apply_chat_template`, `message.get`, `lines.append`, `join` |
| [`_format_sft_batch`](../saddlellm/PeftSFTTrainer.py#L120)<br><sub>`_format_sft_batch(examples: Dict, tokenizer) -> Dict[str, List[str]]`</sub> | function | 模块级格式化批次的内部辅助逻辑。 | `len`, `next`, `iter`, `examples.values`, `range`, `examples.items`, `PostTrainingDataAdapter.normalize_record`, `normalized.get`, `texts.append`, `str` |
| [`_maybe_quantization_config`](../saddlellm/PeftSFTTrainer.py#L143)<br><sub>`_maybe_quantization_config(config: SFTTrainConfig, device: str)`</sub> | function | 模块级实现配置的内部辅助逻辑。 | `RuntimeError`, `BitsAndBytesConfig`, `_supports_bf16` |
| [`_make_training_args`](../saddlellm/PeftSFTTrainer.py#L167)<br><sub>`_make_training_args(config: SFTTrainConfig, device: str)`</sub> | function | 模块级实现训练的内部辅助逻辑。 | `_supports_bf16`, `instantiate_supported` |
| [`_build_sft_trainer`](../saddlellm/PeftSFTTrainer.py#L215)<br><sub>`_build_sft_trainer(**kwargs)`</sub> | function | 模块级构建`build_sft_trainer`的内部辅助逻辑。 | `inspect.signature`, `set`, `kwargs.pop`, `SFTTrainer`, `supported_kwargs` |
| [`generate_response`](../saddlellm/PeftSFTTrainer.py#L224)<br><sub>`generate_response(model, tokenizer, prompt, device: Optional[str]=None, max_new_tokens: int=256) -> str`</sub> | function | Generate a short validation response without assuming CUDA. | `str`, `next`, `model.parameters`, `to`, `tokenizer`, `device.startswith`, `_supports_bf16`, `torch.autocast`, `model.generate`, `tokenizer.decode` |
| [`train_model`](../saddlellm/PeftSFTTrainer.py#L242)<br><sub>`train_model(model_path: str, dataset_path: str, output_path: str, use_lora: bool=True, use_qlora: bool=True, lora_r: int=8, lora_alpha: int=16, lora_dropout: float=0.05, lora_target_modules: Optional[List[str]]=None, learning_rate: float=0.0005, batch_size: int=1, num_epochs: int=3, max_steps: int=-1, save_steps: int=100, max_seq_length: int=512, gradient_accumulation_steps: int=4, warmup_steps: int=100, eval_steps: int=0, logging_steps: int=10, test_prompts: Optional[List[str]]=None, local_files_only: bool=False, trust_remote_code: bool=True, validation_split: float=0.0, response_template: Optional[str]='### Answer:', gradient_checkpointing: bool=True, optim: Optional[str]=None, report_to: str='none', resume_from_checkpoint: Optional[Union[bool, str]]=None)`</sub> | function | 模块级训练模型的公开操作。 | `SFTTrainConfig`, `train_sft` |
| [`train_sft`](../saddlellm/PeftSFTTrainer.py#L305)<br><sub>`train_sft(config: SFTTrainConfig)`</sub> | function | 模块级训练`train_sft`的公开操作。 | `_device`, `logger.info`, `stabilize_peft_optional_backends`, `logger.warning`, `load_tokenizer_compatible`, `_load_sft_dataset`, `isinstance`, `ValueError`, `dataset.map`, `_format_sft_batch` |
| [`train_sft.ValidationCallback.on_step_end`](../saddlellm/PeftSFTTrainer.py#L399)<br><sub>`on_step_end(self, args, state, control, **kwargs)`</sub> | nested function | `ValidationCallback` 中实现`on_step_end`的局部回调/辅助逻辑。 | `model.eval`, `generate_response`, `logger.info`, `len`, `model.train` |
| [`parse_args`](../saddlellm/PeftSFTTrainer.py#L426)<br><sub>`parse_args()`</sub> | function | 模块级解析`parse_args`的公开操作。 | `argparse.ArgumentParser`, `parser.add_argument`, `parser.set_defaults`, `parser.parse_args` |

## `saddlellm/PostTrainingCompatibility.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`installed_post_training_versions`](../saddlellm/PostTrainingCompatibility.py#L30)<br><sub>`installed_post_training_versions() -> Dict[str, str \| None]`</sub> | function | Return installed versions without importing large ML packages. | `metadata.version` |
| [`post_training_runtime_report`](../saddlellm/PostTrainingCompatibility.py#L41)<br><sub>`post_training_runtime_report() -> Dict[str, Any]`</sub> | function | Describe whether the tested post-training dependency set is active. | `installed_post_training_versions`, `POST_TRAINING_REQUIREMENTS.items`, `issues.append`, `SpecifierSet`, `dict` |
| [`format_post_training_runtime_error`](../saddlellm/PostTrainingCompatibility.py#L61)<br><sub>`format_post_training_runtime_error(feature: str, error: BaseException) -> str`</sub> | function | Create an actionable error for lazy TRL/Transformers import failures. | `post_training_runtime_report`, `join` |
| [`callable_accepts_var_kwargs`](../saddlellm/PostTrainingCompatibility.py#L72)<br><sub>`callable_accepts_var_kwargs(callable_obj: Any) -> bool`</sub> | function | Whether a callable explicitly accepts arbitrary keyword arguments. | `any`, `parameters.values`, `inspect.signature` |
| [`supported_kwargs`](../saddlellm/PostTrainingCompatibility.py#L80)<br><sub>`supported_kwargs(callable_obj: Any, values: Mapping[str, Any]) -> Dict[str, Any]`</sub> | function | Drop unsupported/None kwargs unless the callable exposes ``**kwargs``. | `values.items`, `callable_accepts_var_kwargs`, `set`, `inspect.signature`, `cleaned.items` |
| [`instantiate_supported`](../saddlellm/PostTrainingCompatibility.py#L95)<br><sub>`instantiate_supported(config_class: Any, values: Mapping[str, Any])`</sub> | function | Instantiate a version-varying config class with supported keywords. | `config_class`, `supported_kwargs` |
| [`stabilize_peft_optional_backends`](../saddlellm/PostTrainingCompatibility.py#L100)<br><sub>`stabilize_peft_optional_backends() -> List[str]`</sub> | function | Disable installed-but-broken optional PEFT quantization dispatchers. | `list`, `importlib.util.find_spec`, `importlib.import_module`, `metadata.version`, `warnings.append`, `type` |

## `saddlellm/PostTrainingData.py`

共 19 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`NormalizationReport.to_dict`](../saddlellm/PostTrainingData.py#L41)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `NormalizationReport` 转为可序列化字典。 | `asdict` |
| [`PostTrainingDataAdapter.normalize_record`](../saddlellm/PostTrainingData.py#L49)<br><sub>`normalize_record(cls, record: Dict, task: str='sft', source_format: str='auto', keep_metadata: bool=True) -> Optional[Dict]`</sub> | method | `PostTrainingDataAdapter` 中规范化`normalize_record`的公开操作。 | `lower`, `cls.normalize_preference_record`, `cls.normalize_kto_record`, `cls.normalize_rl_record`, `cls.normalize_sft_record` |
| [`PostTrainingDataAdapter.normalize_sft_record`](../saddlellm/PostTrainingData.py#L66)<br><sub>`normalize_sft_record(cls, record: Dict, source_format: str='auto', keep_metadata: bool=True) -> Optional[Dict]`</sub> | method | `PostTrainingDataAdapter` 中规范化`normalize_sft_record`的公开操作。 | `cls._extract_messages`, `cls._has_assistant`, `record.get`, `str`, `cls._first`, `strip`, `cls._attach_metadata` |
| [`PostTrainingDataAdapter.normalize_preference_record`](../saddlellm/PostTrainingData.py#L98)<br><sub>`normalize_preference_record(cls, record: Dict, source_format: str='auto', keep_metadata: bool=True) -> Optional[Dict]`</sub> | method | `PostTrainingDataAdapter` 中规范化`normalize_preference_record`的公开操作。 | `cls._prompt_from_record`, `cls._completion_text`, `cls._first_value`, `cls._attach_metadata` |
| [`PostTrainingDataAdapter.normalize_kto_record`](../saddlellm/PostTrainingData.py#L117)<br><sub>`normalize_kto_record(cls, record: Dict, source_format: str='auto', keep_metadata: bool=True) -> Optional[Dict]`</sub> | method | `PostTrainingDataAdapter` 中规范化`normalize_kto_record`的公开操作。 | `cls._prompt_from_record`, `cls._completion_text`, `cls._first_value`, `record.get`, `isinstance`, `label.lower`, `bool`, `cls._attach_metadata` |
| [`PostTrainingDataAdapter.normalize_rl_record`](../saddlellm/PostTrainingData.py#L140)<br><sub>`normalize_rl_record(cls, record: Dict, source_format: str='auto', keep_metadata: bool=True) -> Optional[Dict]`</sub> | method | `PostTrainingDataAdapter` 中规范化`normalize_rl_record`的公开操作。 | `cls._prompt_from_record`, `cls._first`, `record.get`, `isinstance`, `dict`, `cls._attach_metadata` |
| [`PostTrainingDataAdapter.normalize_records`](../saddlellm/PostTrainingData.py#L158)<br><sub>`normalize_records(cls, records: Iterable[Dict], task: str='sft', source_format: str='auto', keep_metadata: bool=True) -> (List[Dict], NormalizationReport)`</sub> | method | `PostTrainingDataAdapter` 中规范化`normalize_records`的公开操作。 | `NormalizationReport`, `cls.detect_schema`, `report.detected_schemas.get`, `cls.normalize_record`, `normalized.append`, `report.warnings.append` |
| [`PostTrainingDataAdapter.normalize_file`](../saddlellm/PostTrainingData.py#L182)<br><sub>`normalize_file(cls, input_path: str, output_path: str, task: str='sft', source_format: str='auto', keep_metadata: bool=True) -> NormalizationReport`</sub> | method | `PostTrainingDataAdapter` 中规范化`normalize_file`的公开操作。 | `cls.load_records`, `cls.normalize_records`, `os.makedirs`, `os.path.dirname`, `open`, `f.write`, `json.dumps`, `json.dump`, `report.to_dict` |
| [`PostTrainingDataAdapter.load_records`](../saddlellm/PostTrainingData.py#L204)<br><sub>`load_records(path: str) -> List[Dict]`</sub> | method | `PostTrainingDataAdapter` 中加载`load_records`的公开操作。 | `lower`, `os.path.splitext`, `open`, `line.strip`, `records.append`, `json.loads`, `json.load`, `isinstance`, `data.get`, `list` |
| [`PostTrainingDataAdapter.detect_schema`](../saddlellm/PostTrainingData.py#L232)<br><sub>`detect_schema(record: Dict) -> str`</sub> | method | `PostTrainingDataAdapter` 中检测`detect_schema`的公开操作。 | — |
| [`PostTrainingDataAdapter._extract_messages`](../saddlellm/PostTrainingData.py#L248)<br><sub>`_extract_messages(cls, record: Dict) -> List[Dict[str, str]]`</sub> | method | `PostTrainingDataAdapter` 中提取`extract_messages`的内部辅助逻辑。 | `record.get`, `isinstance`, `message.get`, `ROLE_ALIASES.get`, `lower`, `str`, `strip`, `messages.append` |
| [`PostTrainingDataAdapter._prompt_from_record`](../saddlellm/PostTrainingData.py#L265)<br><sub>`_prompt_from_record(cls, record: Dict) -> str`</sub> | method | `PostTrainingDataAdapter` 中记录提示词的内部辅助逻辑。 | `cls._first`, `str`, `cls._extract_messages`, `prompt_messages.append`, `strip`, `join` |
| [`PostTrainingDataAdapter._completion_text`](../saddlellm/PostTrainingData.py#L280)<br><sub>`_completion_text(cls, value) -> Optional[str]`</sub> | method | `PostTrainingDataAdapter` 中实现`completion_text`的内部辅助逻辑。 | `isinstance`, `cls._extract_messages`, `join`, `str`, `value.get`, `json.dumps` |
| [`PostTrainingDataAdapter._first`](../saddlellm/PostTrainingData.py#L297)<br><sub>`_first(record: Dict, keys: Sequence[str]) -> str`</sub> | method | `PostTrainingDataAdapter` 中实现`first`的内部辅助逻辑。 | `PostTrainingDataAdapter._first_value`, `str` |
| [`PostTrainingDataAdapter._first_value`](../saddlellm/PostTrainingData.py#L302)<br><sub>`_first_value(record: Dict, keys: Sequence[str])`</sub> | method | `PostTrainingDataAdapter` 中实现`first_value`的内部辅助逻辑。 | — |
| [`PostTrainingDataAdapter._has_assistant`](../saddlellm/PostTrainingData.py#L309)<br><sub>`_has_assistant(messages: Sequence[Dict[str, str]]) -> bool`</sub> | method | `PostTrainingDataAdapter` 中实现`has_assistant`的内部辅助逻辑。 | `any`, `message.get` |
| [`PostTrainingDataAdapter._attach_metadata`](../saddlellm/PostTrainingData.py#L313)<br><sub>`_attach_metadata(normalized: Dict, record: Dict, task: str, source_format: str, keep_metadata: bool) -> Dict`</sub> | method | `PostTrainingDataAdapter` 中实现`attach_metadata`的内部辅助逻辑。 | `PostTrainingDataAdapter.detect_schema`, `isinstance`, `record.get`, `dict` |
| [`normalize_post_training_record`](../saddlellm/PostTrainingData.py#L326)<br><sub>`normalize_post_training_record(record: Dict, task: str='sft', source_format: str='auto') -> Optional[Dict]`</sub> | function | 模块级规范化训练的公开操作。 | `PostTrainingDataAdapter.normalize_record` |
| [`normalize_post_training_file`](../saddlellm/PostTrainingData.py#L330)<br><sub>`normalize_post_training_file(input_path: str, output_path: str, task: str='sft', source_format: str='auto') -> Dict`</sub> | function | 模块级规范化训练的公开操作。 | `to_dict`, `PostTrainingDataAdapter.normalize_file` |

## `saddlellm/Preference.py`

共 13 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`_device`](../saddlellm/Preference.py#L64)<br><sub>`_device() -> str`</sub> | function | 模块级实现设备的内部辅助逻辑。 | `torch.cuda.is_available` |
| [`_bf16`](../saddlellm/Preference.py#L68)<br><sub>`_bf16() -> bool`</sub> | function | 模块级实现`bf16`的内部辅助逻辑。 | `torch.cuda.is_available`, `torch.cuda.is_bf16_supported` |
| [`_maybe_quantization_config`](../saddlellm/Preference.py#L72)<br><sub>`_maybe_quantization_config(config: PreferenceTrainConfig, device: str)`</sub> | function | 模块级实现配置的内部辅助逻辑。 | `RuntimeError`, `BitsAndBytesConfig`, `_bf16` |
| [`_load_dataset`](../saddlellm/Preference.py#L96)<br><sub>`_load_dataset(path: str)`</sub> | function | 模块级加载数据集的内部辅助逻辑。 | `os.path.isdir`, `load_dataset`, `lower`, `os.path.splitext` |
| [`_normalize_preference_batch`](../saddlellm/Preference.py#L107)<br><sub>`_normalize_preference_batch(batch, tokenizer, method: str)`</sub> | function | 模块级规范化批次的内部辅助逻辑。 | `len`, `next`, `iter`, `batch.values`, `range`, `batch.items`, `PostTrainingDataAdapter.normalize_record`, `append`, `str`, `bool` |
| [`_trainer_class`](../saddlellm/Preference.py#L134)<br><sub>`_trainer_class(method: str)`</sub> | function | 模块级实现`trainer_class`的内部辅助逻辑。 | `importlib.import_module`, `getattr`, `RuntimeError`, `format_post_training_runtime_error` |
| [`_training_args_class`](../saddlellm/Preference.py#L155)<br><sub>`_training_args_class(method: str)`</sub> | function | 模块级实现训练的内部辅助逻辑。 | `importlib.import_module`, `getattr`, `RuntimeError`, `format_post_training_runtime_error` |
| [`_make_training_args`](../saddlellm/Preference.py#L172)<br><sub>`_make_training_args(config: PreferenceTrainConfig)`</sub> | function | 模块级实现训练的内部辅助逻辑。 | `_training_args_class`, `_bf16`, `_device`, `max`, `instantiate_supported` |
| [`_trainer_parameter_names`](../saddlellm/Preference.py#L209)<br><sub>`_trainer_parameter_names(trainer_cls) -> set`</sub> | function | Find the first concrete constructor signature through wrapper classes. | `inspect.signature`, `parameters.items`, `set` |
| [`_build_trainer`](../saddlellm/Preference.py#L223)<br><sub>`_build_trainer(trainer_cls, **kwargs)`</sub> | function | 模块级构建`build_trainer`的内部辅助逻辑。 | `_trainer_parameter_names`, `kwargs.pop`, `trainer_cls`, `supported_kwargs` |
| [`train_preference`](../saddlellm/Preference.py#L230)<br><sub>`train_preference(config: PreferenceTrainConfig)`</sub> | function | 模块级训练`train_preference`的公开操作。 | `_device`, `stabilize_peft_optional_backends`, `logger.warning`, `load_tokenizer_compatible`, `_load_dataset`, `isinstance`, `ValueError`, `dataset.map`, `_normalize_preference_batch`, `len` |
| [`PreferenceTrainer.__init__`](../saddlellm/Preference.py#L340)<br><sub>`__init__(self, config: PreferenceTrainConfig)`</sub> | method | 初始化 `PreferenceTrainer` 实例及其运行依赖。 | — |
| [`PreferenceTrainer.train`](../saddlellm/Preference.py#L343)<br><sub>`train(self)`</sub> | method | `PreferenceTrainer` 中训练`train`的公开操作。 | `train_preference` |

## `saddlellm/PretrainEvalSuite.py`

共 12 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`EvalTaskSpec.to_dict`](../saddlellm/PretrainEvalSuite.py#L29)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `EvalTaskSpec` 转为可序列化字典。 | `asdict` |
| [`EvalSuiteResult.to_dict`](../saddlellm/PretrainEvalSuite.py#L41)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `EvalSuiteResult` 转为可序列化字典。 | — |
| [`PretrainEvalSuite.__init__`](../saddlellm/PretrainEvalSuite.py#L82)<br><sub>`__init__(self, name: str, tasks: Sequence[EvalTaskSpec])`</sub> | method | 初始化 `PretrainEvalSuite` 实例及其运行依赖。 | `list` |
| [`PretrainEvalSuite.for_domain`](../saddlellm/PretrainEvalSuite.py#L87)<br><sub>`for_domain(cls, domain: str, max_samples: int=500) -> 'PretrainEvalSuite'`</sub> | method | `PretrainEvalSuite` 中实现`for_domain`的公开操作。 | `lower`, `cls.DOMAIN_DEFAULTS.get`, `EvalTaskSpec`, `any`, `cls` |
| [`PretrainEvalSuite.from_dict`](../saddlellm/PretrainEvalSuite.py#L106)<br><sub>`from_dict(cls, data: Dict) -> 'PretrainEvalSuite'`</sub> | method | 从字典解析并创建 `PretrainEvalSuite`。 | `cls`, `data.get`, `EvalTaskSpec` |
| [`PretrainEvalSuite.from_json`](../saddlellm/PretrainEvalSuite.py#L113)<br><sub>`from_json(cls, path: str) -> 'PretrainEvalSuite'`</sub> | method | `PretrainEvalSuite` 中实现`from_json`的公开操作。 | `open`, `cls.from_dict`, `json.load` |
| [`PretrainEvalSuite.to_dict`](../saddlellm/PretrainEvalSuite.py#L117)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `PretrainEvalSuite` 转为可序列化字典。 | `t.to_dict` |
| [`PretrainEvalSuite.save`](../saddlellm/PretrainEvalSuite.py#L120)<br><sub>`save(self, path: str) -> str`</sub> | method | `PretrainEvalSuite` 中保存`save`的公开操作。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump`, `self.to_dict` |
| [`PretrainEvalSuite.run`](../saddlellm/PretrainEvalSuite.py#L126)<br><sub>`run(self, model_path: str, output_dir: str, tokenizer_path: Optional[str]=None, batch_size: int=4, cpu: bool=False, skip_missing_local: bool=True) -> EvalSuiteResult`</sub> | method | `PretrainEvalSuite` 中执行`run`的公开操作。 | `os.makedirs`, `Evaluator`, `self._is_missing_local`, `os.path.join`, `evaluator.evaluate`, `add_references`, `ContaminationDetector`, `to_dict`, `detector.scan_files`, `EvalSuiteResult` |
| [`PretrainEvalSuite.summarize`](../saddlellm/PretrainEvalSuite.py#L190)<br><sub>`summarize(self, results: Dict[str, Dict]) -> Dict[str, float]`</sub> | method | `PretrainEvalSuite` 中汇总`summarize`的公开操作。 | `results.items`, `result.get`, `items`, `isinstance` |
| [`PretrainEvalSuite.compare`](../saddlellm/PretrainEvalSuite.py#L201)<br><sub>`compare(result_paths: Sequence[str], output_path: Optional[str]=None) -> Dict`</sub> | method | `PretrainEvalSuite` 中比较`compare`的公开操作。 | `open`, `json.load`, `runs.append`, `sorted`, `run.get`, `get`, `next`, `isinstance`, `rows.append`, `os.makedirs` |
| [`PretrainEvalSuite._is_missing_local`](../saddlellm/PretrainEvalSuite.py#L224)<br><sub>`_is_missing_local(self, dataset: str) -> bool`</sub> | method | `PretrainEvalSuite` 中实现`is_missing_local`的内部辅助逻辑。 | `bool`, `dataset.startswith`, `os.path.exists` |

## `saddlellm/PretrainExperimentRunner.py`

共 9 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`PretrainExperimentConfig.to_dict`](../saddlellm/PretrainExperimentRunner.py#L32)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `PretrainExperimentConfig` 转为可序列化字典。 | `asdict`, `list`, `self.sources_by_bucket.items` |
| [`ExperimentRunManifest.to_dict`](../saddlellm/PretrainExperimentRunner.py#L53)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ExperimentRunManifest` 转为可序列化字典。 | `asdict` |
| [`ExperimentBundle.to_dict`](../saddlellm/PretrainExperimentRunner.py#L66)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ExperimentBundle` 转为可序列化字典。 | `run.to_dict` |
| [`PretrainExperimentRunner.__init__`](../saddlellm/PretrainExperimentRunner.py#L80)<br><sub>`__init__(self, config: Optional[PretrainExperimentConfig]=None)`</sub> | method | 初始化 `PretrainExperimentRunner` 实例及其运行依赖。 | `PretrainExperimentConfig` |
| [`PretrainExperimentRunner.build_bundle`](../saddlellm/PretrainExperimentRunner.py#L83)<br><sub>`build_bundle(self) -> ExperimentBundle`</sub> | method | `PretrainExperimentRunner` 中构建`build_bundle`的公开操作。 | `os.makedirs`, `self._save_json`, `os.path.join`, `self.config.to_dict`, `list`, `DataMixPlanner.for_domain`, `mix_plan.to_pipeline_config`, `mix_plan.to_dict`, `PretrainEvalSuite.for_domain`, `suite.save` |
| [`PretrainExperimentRunner.run`](../saddlellm/PretrainExperimentRunner.py#L186)<br><sub>`run(self, dry_run: bool=True, limit: Optional[int]=None) -> ExperimentBundle`</sub> | method | `PretrainExperimentRunner` 中执行`run`的公开操作。 | `self.build_bundle`, `len`, `open`, `json.load`, `get`, `cfg.get`, `TrainingOrchestrator.from_dict`, `orchestrator.run`, `PretrainReport.generate`, `self._save_json` |
| [`PretrainExperimentRunner.compare_reports`](../saddlellm/PretrainExperimentRunner.py#L207)<br><sub>`compare_reports(self, bundle: Optional[ExperimentBundle]=None) -> Dict`</sub> | method | `PretrainExperimentRunner` 中比较`compare_reports`的公开操作。 | `self.build_bundle`, `os.path.exists`, `reports.append`, `open`, `json.load`, `rows.append`, `report.get`, `len`, `self._save_json`, `os.path.join` |
| [`PretrainExperimentRunner._save_json`](../saddlellm/PretrainExperimentRunner.py#L227)<br><sub>`_save_json(self, path: str, data: Dict) -> str`</sub> | method | `PretrainExperimentRunner` 中保存`save_json`的内部辅助逻辑。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump` |
| [`PretrainExperimentRunner._format_multiplier`](../saddlellm/PretrainExperimentRunner.py#L233)<br><sub>`_format_multiplier(self, target_tokens: int, active_params: int) -> str`</sub> | method | `PretrainExperimentRunner` 中格式化`format_multiplier`的内部辅助逻辑。 | `max`, `replace` |

## `saddlellm/PretrainReport.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ReportFinding.to_dict`](../saddlellm/PretrainReport.py#L14)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `ReportFinding` 转为可序列化字典。 | `asdict` |
| [`PretrainReportResult.to_dict`](../saddlellm/PretrainReport.py#L27)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `PretrainReportResult` 转为可序列化字典。 | `f.to_dict` |
| [`PretrainReportResult.to_markdown`](../saddlellm/PretrainReport.py#L37)<br><sub>`to_markdown(self) -> str`</sub> | method | `PretrainReportResult` 中实现`to_markdown`的公开操作。 | `lines.append`, `item.severity.upper`, `self.sections.items`, `lines.extend`, `title`, `name.replace`, `section.items`, `isinstance`, `json.dumps`, `len` |
| [`PretrainReport.from_output_dir`](../saddlellm/PretrainReport.py#L83)<br><sub>`from_output_dir(cls, output_dir: str, extra_artifacts: Optional[Dict[str, str]]=None) -> PretrainReportResult`</sub> | method | `PretrainReport` 中实现输出的公开操作。 | `cls._discover`, `cls._load_json`, `artifacts.items`, `os.path.exists`, `cls._findings`, `cls._score`, `PretrainReportResult`, `cls._grade` |
| [`PretrainReport.save`](../saddlellm/PretrainReport.py#L98)<br><sub>`save(cls, output_dir: str, report: PretrainReportResult, json_name: str='pretrain_report.json', markdown_name: str='pretrain_report.md') -> Dict[str, str]`</sub> | method | `PretrainReport` 中保存`save`的公开操作。 | `os.makedirs`, `os.path.join`, `open`, `json.dump`, `report.to_dict`, `f.write`, `report.to_markdown` |
| [`PretrainReport.generate`](../saddlellm/PretrainReport.py#L115)<br><sub>`generate(cls, output_dir: str, extra_artifacts: Optional[Dict[str, str]]=None) -> PretrainReportResult`</sub> | method | `PretrainReport` 中生成`generate`的公开操作。 | `cls.from_output_dir`, `cls.save` |
| [`PretrainReport._discover`](../saddlellm/PretrainReport.py#L121)<br><sub>`_discover(cls, output_dir: str, extra: Dict[str, str]) -> Dict[str, str]`</sub> | method | `PretrainReport` 中实现`discover`的内部辅助逻辑。 | `cls.DEFAULT_FILES.items`, `os.path.isabs`, `os.path.join`, `os.path.exists`, `extra.items` |
| [`PretrainReport._load_json`](../saddlellm/PretrainReport.py#L133)<br><sub>`_load_json(path: str) -> Dict`</sub> | method | `PretrainReport` 中加载`load_json`的内部辅助逻辑。 | `open`, `json.load`, `str` |
| [`PretrainReport._findings`](../saddlellm/PretrainReport.py#L141)<br><sub>`_findings(cls, sections: Dict[str, Dict]) -> List[ReportFinding]`</sub> | method | `PretrainReport` 中实现`findings`的内部辅助逻辑。 | `sections.get`, `tok.get`, `findings.append`, `ReportFinding`, `stability.get`, `len`, `any`, `e.get`, `model_metrics.get`, `arch.get` |
| [`PretrainReport._score`](../saddlellm/PretrainReport.py#L202)<br><sub>`_score(findings: List[ReportFinding], sections: Dict[str, Dict]) -> float`</sub> | method | `PretrainReport` 中评分`score`的内部辅助逻辑。 | `max`, `min` |
| [`PretrainReport._grade`](../saddlellm/PretrainReport.py#L216)<br><sub>`_grade(score: float) -> str`</sub> | method | `PretrainReport` 中实现`grade`的内部辅助逻辑。 | — |

## `saddlellm/PretrainStability.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`StabilityEvent.to_dict`](../saddlellm/PretrainStability.py#L29)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `StabilityEvent` 转为可序列化字典。 | `asdict` |
| [`PretrainStabilityCallback.__init__`](../saddlellm/PretrainStability.py#L42)<br><sub>`__init__(self, config: Optional[StabilityConfig]=None)`</sub> | method | 初始化 `PretrainStabilityCallback` 实例及其运行依赖。 | `StabilityConfig`, `os.makedirs`, `os.path.join` |
| [`PretrainStabilityCallback.on_log`](../saddlellm/PretrainStability.py#L50)<br><sub>`on_log(self, args, state, control, logs=None, **kwargs)`</sub> | method | `PretrainStabilityCallback` 中记录`on_log`的公开操作。 | `getattr`, `int`, `logs.get`, `float`, `self._jsonable`, `logs.items`, `self._check_loss`, `self._record`, `self._write_summary` |
| [`PretrainStabilityCallback.on_train_end`](../saddlellm/PretrainStability.py#L71)<br><sub>`on_train_end(self, args, state, control, **kwargs)`</sub> | method | `PretrainStabilityCallback` 中训练`on_train_end`的公开操作。 | `getattr`, `self._write_summary` |
| [`PretrainStabilityCallback.summary`](../saddlellm/PretrainStability.py#L77)<br><sub>`summary(self) -> Dict`</sub> | method | `PretrainStabilityCallback` 中实现`summary`的公开操作。 | `math.isfinite`, `len`, `e.to_dict`, `min`, `self._trend` |
| [`PretrainStabilityCallback._check_loss`](../saddlellm/PretrainStability.py#L88)<br><sub>`_check_loss(self, step: int, loss: float, metrics: Dict) -> Optional[StabilityEvent]`</sub> | method | `PretrainStabilityCallback` 中检查`check_loss`的内部辅助逻辑。 | `self.loss_history.append`, `math.isnan`, `math.isinf`, `StabilityEvent`, `len`, `max`, `math.isfinite`, `sum` |
| [`PretrainStabilityCallback._record`](../saddlellm/PretrainStability.py#L130)<br><sub>`_record(self, event: StabilityEvent)`</sub> | method | `PretrainStabilityCallback` 中记录`record`的内部辅助逻辑。 | `self.events.append`, `open`, `f.write`, `json.dumps`, `event.to_dict` |
| [`PretrainStabilityCallback._write_summary`](../saddlellm/PretrainStability.py#L136)<br><sub>`_write_summary(self)`</sub> | method | `PretrainStabilityCallback` 中写入`write_summary`的内部辅助逻辑。 | `open`, `json.dump`, `self.summary` |
| [`PretrainStabilityCallback._trend`](../saddlellm/PretrainStability.py#L140)<br><sub>`_trend(self, losses: List[float]) -> str`</sub> | method | `PretrainStabilityCallback` 中实现`trend`的内部辅助逻辑。 | `len`, `max`, `sum` |
| [`PretrainStabilityCallback._jsonable`](../saddlellm/PretrainStability.py#L152)<br><sub>`_jsonable(self, value)`</sub> | method | `PretrainStabilityCallback` 中实现`jsonable`的内部辅助逻辑。 | `json.dumps`, `str` |
| [`CheckpointInspector.inspect`](../saddlellm/PretrainStability.py#L168)<br><sub>`inspect(cls, checkpoint_dir: str) -> Dict`</sub> | method | `CheckpointInspector` 中检查`inspect`的公开操作。 | `os.path.isdir`, `os.path.exists`, `os.path.join`, `missing.append`, `any`, `warnings.append` |

## `saddlellm/PromptTuner.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`PromptTuner.__init__`](../saddlellm/PromptTuner.py#L34)<br><sub>`__init__(self, model_name: str='meta-llama/Meta-Llama-3-8B', method: Literal['soft', 'prefix']='soft', num_virtual_tokens: int=20, max_length: int=1024, device_map: str='auto', chat_template: str='default')`</sub> | method | Initialize Prompt Tuner. | `AutoModelForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained`, `self.set_chat_template`, `PromptTuningConfig`, `PrefixTuningConfig`, `get_peft_model`, `self.model.print_trainable_parameters` |
| [`PromptTuner.set_chat_template`](../saddlellm/PromptTuner.py#L84)<br><sub>`set_chat_template(self, template_type: str)`</sub> | method | Set chat template (same as RLHF example) | — |
| [`PromptTuner.fit`](../saddlellm/PromptTuner.py#L93)<br><sub>`fit(self, train_dataset: Dataset, eval_dataset: Optional[Dataset]=None, epochs: int=3, batch_size: int=2, learning_rate: float=0.03, output_dir: str='./prompt_tuning_output', logging_steps: int=10)`</sub> | method | Run prompt tuning training. | `train_dataset.map`, `eval_dataset.map`, `TrainingArguments`, `torch.cuda.is_bf16_supported`, `DataCollatorForLanguageModeling`, `Trainer`, `trainer.train` |
| [`PromptTuner.fit.tokenize_fn`](../saddlellm/PromptTuner.py#L106)<br><sub>`tokenize_fn(examples: Dict) -> Dict`</sub> | nested function | `PromptTuner` 中分词`tokenize_fn`的局部回调/辅助逻辑。 | `self.tokenizer` |
| [`PromptTuner.generate`](../saddlellm/PromptTuner.py#L150)<br><sub>`generate(self, prompt: str, max_new_tokens: int=100) -> str`</sub> | method | Generate text with learned prompts. | `to`, `self.tokenizer`, `self.model.generate`, `self.tokenizer.decode` |
| [`PromptTuner.save`](../saddlellm/PromptTuner.py#L166)<br><sub>`save(self, path: str)`</sub> | method | Save only prompt embeddings (tiny files). | `self.model.save_pretrained`, `self.tokenizer.save_pretrained` |
| [`PromptTuner.load`](../saddlellm/PromptTuner.py#L172)<br><sub>`load(cls, path: str, **kwargs)`</sub> | method | Load trained prompt tuner. | `cls` |

## `saddlellm/PublicPolicyRLVR.py`

共 36 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`PolicyExample.primary_label`](../saddlellm/PublicPolicyRLVR.py#L69)<br><sub>`primary_label(self) -> str`</sub> | method | `PolicyExample` 中实现`primary_label`的公开操作。 | — |
| [`PublicPolicyPilotConfig.validate`](../saddlellm/PublicPolicyRLVR.py#L121)<br><sub>`validate(self) -> None`</sub> | method | `PublicPolicyPilotConfig` 中校验`validate`的公开操作。 | `logger.warning`, `ValueError`, `getattr` |
| [`PublicPolicyPilotConfig.from_yaml`](../saddlellm/PublicPolicyRLVR.py#L139)<br><sub>`from_yaml(cls, path: Union[os.PathLike, str]) -> 'PublicPolicyPilotConfig'`</sub> | method | `PublicPolicyPilotConfig` 中实现`from_yaml`的公开操作。 | `open`, `yaml.safe_load`, `set`, `ValueError`, `sorted`, `cls`, `config.validate` |
| [`_parse_problem`](../saddlellm/PublicPolicyRLVR.py#L154)<br><sub>`_parse_problem(value: Any) -> Dict[str, Any]`</sub> | function | 模块级解析`parse_problem`的内部辅助逻辑。 | `isinstance`, `TypeError`, `type`, `json.loads`, `ValueError` |
| [`_normalize_question_id`](../saddlellm/PublicPolicyRLVR.py#L165)<br><sub>`_normalize_question_id(problem: Mapping[str, Any]) -> str`</sub> | function | 模块级规范化`normalize_question_id`的内部辅助逻辑。 | `upper`, `strip`, `str`, `problem.get`, `re.search`, `int`, `match.group`, `ValueError` |
| [`_normalize_organization`](../saddlellm/PublicPolicyRLVR.py#L177)<br><sub>`_normalize_organization(value: Any) -> str`</sub> | function | 模块级规范化`normalize_organization`的内部辅助逻辑。 | `casefold`, `re.sub`, `strip`, `str` |
| [`_case_key`](../saddlellm/PublicPolicyRLVR.py#L181)<br><sub>`_case_key(problem: Mapping[str, Any]) -> str`</sub> | function | 模块级实现`case_key`的内部辅助逻辑。 | `_normalize_question_id`, `_normalize_organization`, `problem.get` |
| [`_stable_bucket`](../saddlellm/PublicPolicyRLVR.py#L185)<br><sub>`_stable_bucket(case_key: str, seed: int, buckets: int=10000) -> int`</sub> | function | 模块级实现`stable_bucket`的内部辅助逻辑。 | `hexdigest`, `hashlib.sha256`, `encode`, `int` |
| [`split_for_case`](../saddlellm/PublicPolicyRLVR.py#L190)<br><sub>`split_for_case(case_key: str, seed: int, train_fraction: float, validation_fraction: float) -> str`</sub> | function | 模块级切分`split_for_case`的公开操作。 | `_stable_bucket` |
| [`AASBDataModule.__init__`](../saddlellm/PublicPolicyRLVR.py#L207)<br><sub>`__init__(self, config: PublicPolicyPilotConfig)`</sub> | method | 初始化 `AASBDataModule` 实例及其运行依赖。 | — |
| [`AASBDataModule.load`](../saddlellm/PublicPolicyRLVR.py#L213)<br><sub>`load(self) -> 'AASBDataModule'`</sub> | method | `AASBDataModule` 中加载`load`的公开操作。 | `load_dataset`, `defaultdict`, `dataset_b.items`, `_parse_problem`, `_case_key`, `isinstance`, `tuple`, `map`, `add`, `dataset_a.items` |
| [`AASBDataModule.split_of`](../saddlellm/PublicPolicyRLVR.py#L343)<br><sub>`split_of(self, example: PolicyExample) -> str`</sub> | method | `AASBDataModule` 中切分`split_of`的公开操作。 | `split_for_case` |
| [`AASBDataModule._derived_keys`](../saddlellm/PublicPolicyRLVR.py#L351)<br><sub>`_derived_keys(self, examples: Iterable[PolicyExample]) -> Dict[str, set[str]]`</sub> | method | `AASBDataModule` 中实现`derived_keys`的内部辅助逻辑。 | `defaultdict`, `add`, `self.split_of` |
| [`AASBDataModule.select`](../saddlellm/PublicPolicyRLVR.py#L357)<br><sub>`select(self, task: str, split: str, limit: int, balanced: bool=False) -> List[PolicyExample]`</sub> | method | `AASBDataModule` 中实现`select`的公开操作。 | `self.split_of`, `candidates.sort`, `_stable_bucket`, `defaultdict`, `append`, `sorted`, `len`, `selected.append`, `pop`, `next_labels.append` |
| [`_b_label_sort_key`](../saddlellm/PublicPolicyRLVR.py#L390)<br><sub>`_b_label_sort_key(label: str) -> Tuple[int, str]`</sub> | function | 模块级实现`b_label_sort_key`的内部辅助逻辑。 | — |
| [`_abbreviate_evidence`](../saddlellm/PublicPolicyRLVR.py#L394)<br><sub>`_abbreviate_evidence(text: str, max_chars: Optional[int]) -> str`</sub> | function | 模块级实现`abbreviate_evidence`的内部辅助逻辑。 | `len`, `int` |
| [`build_user_prompt`](../saddlellm/PublicPolicyRLVR.py#L404)<br><sub>`build_user_prompt(example: PolicyExample, use_cot: bool=False, max_response_chars: Optional[int]=None) -> str`</sub> | function | 模块级构建提示词的公开操作。 | `_abbreviate_evidence`, `join`, `sorted`, `example.candidate_themes.items` |
| [`render_chat_prompt`](../saddlellm/PublicPolicyRLVR.py#L445)<br><sub>`render_chat_prompt(tokenizer, user_prompt: str, enable_thinking: bool=False) -> str`</sub> | function | 模块级渲染提示词的公开操作。 | `dict`, `tokenizer.apply_chat_template` |
| [`extract_answer`](../saddlellm/PublicPolicyRLVR.py#L472)<br><sub>`extract_answer(response: str, task: str) -> Tuple[str, ...]`</sub> | function | 模块级提取`extract_answer`的公开操作。 | `_ANSWER_TAG.search`, `tagged.group`, `casefold`, `content.strip`, `strip`, `re.sub`, `sorted`, `replace`, `re.escape`, `re.search` |
| [`build_verifiable_reward`](../saddlellm/PublicPolicyRLVR.py#L512)<br><sub>`build_verifiable_reward(prompt_examples: Mapping[str, PolicyExample], correctness_weight: float=0.9, format_weight: float=0.1)`</sub> | function | 模块级构建`build_verifiable_reward`的公开操作。 | `math_is_close`, `ValueError` |
| [`build_verifiable_reward.reward`](../saddlellm/PublicPolicyRLVR.py#L520)<br><sub>`reward(prompt: str, response: str) -> float`</sub> | nested function | 模块级实现`reward`的局部回调/辅助逻辑。 | `extract_answer`, `float`, `_ANSWER_TAG.search`, `bool` |
| [`math_is_close`](../saddlellm/PublicPolicyRLVR.py#L530)<br><sub>`math_is_close(left: float, right: float, tolerance: float=1e-09) -> bool`</sub> | function | 模块级关闭`math_is_close`的公开操作。 | `abs` |
| [`_macro_f1_single`](../saddlellm/PublicPolicyRLVR.py#L534)<br><sub>`_macro_f1_single(expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> float`</sub> | function | 模块级实现`macro_f1_single`的内部辅助逻辑。 | `sum`, `zip`, `scores.append`, `len` |
| [`_set_f1`](../saddlellm/PublicPolicyRLVR.py#L547)<br><sub>`_set_f1(expected: Sequence[str], predicted: Sequence[str]) -> float`</sub> | function | 模块级设置`set_f1`的内部辅助逻辑。 | `set`, `len` |
| [`_jaccard`](../saddlellm/PublicPolicyRLVR.py#L555)<br><sub>`_jaccard(expected: Sequence[str], predicted: Sequence[str]) -> float`</sub> | function | 模块级实现`jaccard`的内部辅助逻辑。 | `set`, `len` |
| [`compute_metrics`](../saddlellm/PublicPolicyRLVR.py#L561)<br><sub>`compute_metrics(examples: Sequence[PolicyExample], predictions: Sequence[Sequence[Tuple[str, ...]]], format_validity: Optional[Sequence[Sequence[bool]]]=None) -> Dict[str, float]`</sub> | function | 模块级计算指标的公开操作。 | `len`, `ValueError`, `any`, `zip`, `float`, `exact_values.extend`, `pass_values.append`, `invalid_values.extend`, `consistency_values.append`, `max` |
| [`PolicyBenchmarkEvaluator.__init__`](../saddlellm/PublicPolicyRLVR.py#L629)<br><sub>`__init__(self, model, tokenizer, config: PublicPolicyPilotConfig)`</sub> | method | 初始化 `PolicyBenchmarkEvaluator` 实例及其运行依赖。 | `next`, `model.parameters` |
| [`PolicyBenchmarkEvaluator.evaluate`](../saddlellm/PublicPolicyRLVR.py#L642)<br><sub>`evaluate(self, examples: Sequence[PolicyExample], output_path: Optional[Union[os.PathLike, str]]=None) -> Dict[str, Any]`</sub> | method | `PolicyBenchmarkEvaluator` 中评估`evaluate`的公开操作。 | `self.model.eval`, `torch.manual_seed`, `torch.cuda.is_available`, `torch.cuda.manual_seed_all`, `build_user_prompt`, `render_chat_prompt`, `to`, `self.tokenizer`, `generation_kwargs.update`, `torch.inference_mode` |
| [`load_qwen3_lora`](../saddlellm/PublicPolicyRLVR.py#L735)<br><sub>`load_qwen3_lora(config: PublicPolicyPilotConfig)`</sub> | function | Load Qwen3 and attach a small LoRA policy adapter. | `Version`, `RuntimeError`, `torch.cuda.is_available`, `AutoTokenizer.from_pretrained`, `to`, `AutoModelForCausalLM.from_pretrained`, `callable`, `getattr`, `model.gradient_checkpointing_enable`, `model.enable_input_require_grads` |
| [`prepare_pilot_data`](../saddlellm/PublicPolicyRLVR.py#L815)<br><sub>`prepare_pilot_data(config: PublicPolicyPilotConfig) -> Dict[str, Any]`</sub> | function | 模块级准备数据的公开操作。 | `load`, `AASBDataModule`, `data.select` |
| [`_example_manifest`](../saddlellm/PublicPolicyRLVR.py#L830)<br><sub>`_example_manifest(example: PolicyExample, split: str) -> Dict[str, Any]`</sub> | function | 模块级实现数据清单的内部辅助逻辑。 | `list` |
| [`dry_run_public_policy_pilot`](../saddlellm/PublicPolicyRLVR.py#L841)<br><sub>`dry_run_public_policy_pilot(config: PublicPolicyPilotConfig) -> Dict[str, Any]`</sub> | function | Validate data, split isolation, prompts, parsers, and rewards without a model. | `config.validate`, `prepare_pilot_data`, `build_user_prompt`, `build_verifiable_reward`, `prompt_examples.items`, `join`, `reward_checks.append`, `reward`, `all`, `AssertionError` |
| [`run_public_policy_pilot`](../saddlellm/PublicPolicyRLVR.py#L900)<br><sub>`run_public_policy_pilot(config: PublicPolicyPilotConfig) -> Dict[str, Any]`</sub> | function | Execute the minimal Qwen3 baseline -> A-RLVR -> A/B evaluation loop. | `config.validate`, `Path`, `output_dir.mkdir`, `prepare_pilot_data`, `load_qwen3_lora`, `PolicyBenchmarkEvaluator`, `logger.info`, `evaluator.evaluate`, `render_chat_prompt`, `build_user_prompt` |
| [`_metric_delta`](../saddlellm/PublicPolicyRLVR.py#L1060)<br><sub>`_metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, float]`</sub> | function | 模块级实现`metric_delta`的内部辅助逻辑。 | `sorted`, `set`, `isinstance`, `float` |
| [`environment_metadata`](../saddlellm/PublicPolicyRLVR.py#L1070)<br><sub>`environment_metadata() -> Dict[str, Any]`</sub> | function | 模块级实现环境的公开操作。 | `torch.cuda.is_available`, `torch.cuda.get_device_properties`, `platform.platform` |
| [`_write_json`](../saddlellm/PublicPolicyRLVR.py#L1094)<br><sub>`_write_json(path: Union[os.PathLike, str], value: Any) -> None`</sub> | function | 模块级写入`write_json`的内部辅助逻辑。 | `Path`, `destination.parent.mkdir`, `destination.open`, `json.dump` |

## `saddlellm/PublicPolicySFT.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`build_sft_completion`](../saddlellm/PublicPolicySFT.py#L34)<br><sub>`build_sft_completion(example: PolicyExample) -> str`</sub> | function | Render the ground-truth answer as an SFT target completion. | `join` |
| [`build_sft_message_pair`](../saddlellm/PublicPolicySFT.py#L42)<br><sub>`build_sft_message_pair(example: PolicyExample, tokenizer, config: PublicPolicyPilotConfig) -> Dict[str, str]`</sub> | function | Return ``{"prompt": ..., "completion": ...}`` for one example. | `build_user_prompt`, `render_chat_prompt`, `build_sft_completion` |
| [`prepare_sft_dataset`](../saddlellm/PublicPolicySFT.py#L62)<br><sub>`prepare_sft_dataset(examples: Sequence[PolicyExample], tokenizer, config: PublicPolicyPilotConfig) -> Dataset`</sub> | function | Build a HuggingFace Dataset of (prompt + completion) texts. | `records.append`, `build_sft_message_pair`, `Dataset.from_list`, `dataset.map` |
| [`prepare_sft_dataset._format_text`](../saddlellm/PublicPolicySFT.py#L72)<br><sub>`_format_text(record: Dict[str, str]) -> Dict[str, str]`</sub> | nested function | 模块级格式化`format_text`的局部回调/辅助逻辑。 | — |
| [`_load_base_model`](../saddlellm/PublicPolicySFT.py#L80)<br><sub>`_load_base_model(config: PublicPolicyPilotConfig)`</sub> | function | Load Qwen3 base model and tokenizer WITHOUT LoRA. | `Version`, `RuntimeError`, `torch.cuda.is_available`, `AutoTokenizer.from_pretrained`, `to`, `AutoModelForCausalLM.from_pretrained`, `sum`, `p.numel`, `model.parameters`, `str` |
| [`SFTExperimentConfig.validate`](../saddlellm/PublicPolicySFT.py#L152)<br><sub>`validate(self) -> None`</sub> | method | `SFTExperimentConfig` 中校验`validate`的公开操作。 | `ValueError` |
| [`train_public_policy_sft`](../saddlellm/PublicPolicySFT.py#L161)<br><sub>`train_public_policy_sft(pilot_config: PublicPolicyPilotConfig, sft_config: SFTExperimentConfig) -> Dict[str, Any]`</sub> | function | Run SFT on the configured source-task training split and return the LoRA adapter path. | `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `logger.info`, `load`, `AASBDataModule`, `data.select`, `_load_base_model`, `len` |

## `saddlellm/RLHFTrainer.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`RLHFTrainer.__init__`](../saddlellm/RLHFTrainer.py#L33)<br><sub>`__init__(self, method: Literal['ppo', 'dpo']='dpo', model_name: str='meta-llama/Meta-Llama-3-8B', reward_model: Optional[str]=None, max_length: int=1024, use_lora: bool=True, lora_rank: int=64, device_map: str='auto', chat_template: str='default')`</sub> | method | Initialize RLHF trainer. | `AutoTokenizer.from_pretrained`, `self.set_chat_template`, `AutoModelForCausalLMWithValueHead.from_pretrained`, `create_reference_model`, `pipeline`, `AutoModelForCausalLM.from_pretrained`, `LoraConfig`, `get_peft_model` |
| [`RLHFTrainer.set_chat_template`](../saddlellm/RLHFTrainer.py#L100)<br><sub>`set_chat_template(self, template_type: str)`</sub> | method | 设置对话模板（同SFT示例） | — |
| [`RLHFTrainer.fit`](../saddlellm/RLHFTrainer.py#L109)<br><sub>`fit(self, dataset: Dataset, epochs: int=3, batch_size: int=2, learning_rate: float=1e-05, output_dir: str='./rlhf_output', beta: float=0.1, kl_penalty: float=0.2)`</sub> | method | Run RLHF training. | `dataset.map`, `TrainingArguments`, `torch.cuda.is_bf16_supported`, `PPOTrainer`, `range`, `input_ids.to`, `self.tokenizer`, `self.trainer.generate`, `torch.tensor`, `self.reward_model` |
| [`RLHFTrainer.fit.tokenize_fn`](../saddlellm/RLHFTrainer.py#L128)<br><sub>`tokenize_fn(examples: Dict) -> Dict`</sub> | nested function | `RLHFTrainer` 中分词`tokenize_fn`的局部回调/辅助逻辑。 | `self.tokenizer`, `self.tokenizer.apply_chat_template` |
| [`RLHFTrainer.generate`](../saddlellm/RLHFTrainer.py#L202)<br><sub>`generate(self, prompt: str, max_new_tokens: int=100) -> str`</sub> | method | 生成文本 | `to`, `self.tokenizer`, `self.model.generate`, `self.tokenizer.decode` |
| [`RLHFTrainer.save`](../saddlellm/RLHFTrainer.py#L218)<br><sub>`save(self, path: str)`</sub> | method | 保存模型（含适配器） | `self.model.save_pretrained`, `self.tokenizer.save_pretrained` |
| [`RLHFTrainer.load`](../saddlellm/RLHFTrainer.py#L224)<br><sub>`load(cls, path: str, **kwargs)`</sub> | method | 加载训练后的模型 | `cls` |

## `saddlellm/RLScaling.py`

共 33 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`RolloutSample.to_sft_record`](../saddlellm/RLScaling.py#L51)<br><sub>`to_sft_record(self) -> Dict[str, str]`</sub> | method | `RolloutSample` 中记录`to_sft_record`的公开操作。 | — |
| [`ActionRoleClassifier.__init__`](../saddlellm/RLScaling.py#L136)<br><sub>`__init__(self, config: Optional[TriageConfig]=None)`</sub> | method | 初始化 `ActionRoleClassifier` 实例及其运行依赖。 | `TriageConfig` |
| [`ActionRoleClassifier.segment`](../saddlellm/RLScaling.py#L139)<br><sub>`segment(self, text: str) -> List[RolloutSegment]`</sub> | method | `ActionRoleClassifier` 中实现`segment`的公开操作。 | `self._split`, `text.find`, `len`, `self.classify`, `segments.append`, `RolloutSegment` |
| [`ActionRoleClassifier.classify`](../saddlellm/RLScaling.py#L153)<br><sub>`classify(self, text: str) -> str`</sub> | method | `ActionRoleClassifier` 中实现`classify`的公开操作。 | `text.lower`, `self.ROLE_PATTERNS.items`, `sum`, `re.search` |
| [`ActionRoleClassifier._split`](../saddlellm/RLScaling.py#L164)<br><sub>`_split(self, text: str) -> List[str]`</sub> | method | `ActionRoleClassifier` 中切分`split`的内部辅助逻辑。 | `p.strip`, `re.split`, `text.splitlines`, `len`, `text.strip` |
| [`TriageCreditAssigner.__init__`](../saddlellm/RLScaling.py#L179)<br><sub>`__init__(self, config: Optional[TriageConfig]=None, role_reward_fn: Optional[Callable[[RolloutSegment, RolloutSample], float]]=None, classifier: Optional[ActionRoleClassifier]=None)`</sub> | method | 初始化 `TriageCreditAssigner` 实例及其运行依赖。 | `TriageConfig`, `ActionRoleClassifier` |
| [`TriageCreditAssigner.assign`](../saddlellm/RLScaling.py#L189)<br><sub>`assign(self, sample: RolloutSample) -> RolloutSample`</sub> | method | `TriageCreditAssigner` 中实现`assign`的公开操作。 | `self.classifier.segment`, `self.config.role_weights.get`, `self.role_reward_fn`, `float`, `credits.append`, `sum`, `max`, `len`, `self._expand_token_credit`, `self.role_histogram` |
| [`TriageCreditAssigner.role_histogram`](../saddlellm/RLScaling.py#L215)<br><sub>`role_histogram(sample: RolloutSample) -> Dict[str, int]`</sub> | method | `TriageCreditAssigner` 中实现`role_histogram`的公开操作。 | `hist.get` |
| [`TriageCreditAssigner._expand_token_credit`](../saddlellm/RLScaling.py#L221)<br><sub>`_expand_token_credit(self, sample: RolloutSample) -> List[float]`</sub> | method | `TriageCreditAssigner` 中实现`expand_token_credit`的内部辅助逻辑。 | `max`, `len`, `re.findall`, `token_credits.extend` |
| [`TriageRolloutProcessor.__init__`](../saddlellm/RLScaling.py#L232)<br><sub>`__init__(self, assigner: Optional[TriageCreditAssigner]=None)`</sub> | method | 初始化 `TriageRolloutProcessor` 实例及其运行依赖。 | `TriageCreditAssigner` |
| [`TriageRolloutProcessor.process`](../saddlellm/RLScaling.py#L235)<br><sub>`process(self, samples: Iterable[RolloutSample]) -> List[RolloutSample]`</sub> | method | `TriageRolloutProcessor` 中处理`process`的公开操作。 | `self.assigner.assign` |
| [`RewardManager.__init__`](../saddlellm/RLScaling.py#L242)<br><sub>`__init__(self, reward_fns: Sequence[RewardFn], weights: Optional[Sequence[float]]=None)`</sub> | method | 初始化 `RewardManager` 实例及其运行依赖。 | `ValueError`, `list`, `len` |
| [`RewardManager.__call__`](../saddlellm/RLScaling.py#L250)<br><sub>`__call__(self, prompt: str, response: str) -> float`</sub> | method | `RewardManager` 中实现`call__`的内部辅助逻辑。 | `zip`, `float`, `fn`, `abs`, `max` |
| [`RewardManager.length_reward`](../saddlellm/RLScaling.py#L259)<br><sub>`length_reward(target_min: int=200, target_max: int=4000) -> RewardFn`</sub> | method | `RewardManager` 中实现`length_reward`的公开操作。 | — |
| [`RewardManager.length_reward._reward`](../saddlellm/RLScaling.py#L260)<br><sub>`_reward(prompt: str, response: str) -> float`</sub> | nested function | `RewardManager` 中实现`reward`的局部回调/辅助逻辑。 | `len`, `max` |
| [`RewardManager.reasoning_format_reward`](../saddlellm/RLScaling.py#L270)<br><sub>`reasoning_format_reward() -> RewardFn`</sub> | method | `RewardManager` 中格式化`reasoning_format_reward`的公开操作。 | — |
| [`RewardManager.reasoning_format_reward._reward`](../saddlellm/RLScaling.py#L276)<br><sub>`_reward(prompt: str, response: str) -> float`</sub> | nested function | `RewardManager` 中实现`reward`的局部回调/辅助逻辑。 | `response.strip`, `any`, `re.search`, `_has_bad_repetition`, `min` |
| [`RewardManager.exact_answer_reward`](../saddlellm/RLScaling.py#L290)<br><sub>`exact_answer_reward(answer_extractor: Callable[[str], str], target_lookup: Dict[str, str]) -> RewardFn`</sub> | method | `RewardManager` 中实现`exact_answer_reward`的公开操作。 | — |
| [`RewardManager.exact_answer_reward._reward`](../saddlellm/RLScaling.py#L291)<br><sub>`_reward(prompt: str, response: str) -> float`</sub> | nested function | `RewardManager` 中实现`reward`的局部回调/辅助逻辑。 | `target_lookup.get`, `answer_extractor`, `strip`, `str` |
| [`RolloutBuffer.__init__`](../saddlellm/RLScaling.py#L303)<br><sub>`__init__(self)`</sub> | method | 初始化 `RolloutBuffer` 实例及其运行依赖。 | — |
| [`RolloutBuffer.add_many`](../saddlellm/RLScaling.py#L306)<br><sub>`add_many(self, samples: Iterable[RolloutSample])`</sub> | method | `RolloutBuffer` 中添加`add_many`的公开操作。 | `self.samples.extend` |
| [`RolloutBuffer.accepted`](../saddlellm/RLScaling.py#L309)<br><sub>`accepted(self) -> List[RolloutSample]`</sub> | method | `RolloutBuffer` 中实现`accepted`的公开操作。 | — |
| [`RolloutBuffer.by_prompt`](../saddlellm/RLScaling.py#L312)<br><sub>`by_prompt(self) -> Dict[str, List[RolloutSample]]`</sub> | method | `RolloutBuffer` 中实现提示词的公开操作。 | `append`, `grouped.setdefault` |
| [`RolloutBuffer.save_jsonl`](../saddlellm/RLScaling.py#L318)<br><sub>`save_jsonl(self, path: str, accepted_only: bool=True) -> str`</sub> | method | `RolloutBuffer` 中保存`save_jsonl`的公开操作。 | `os.makedirs`, `os.path.dirname`, `self.accepted`, `open`, `f.write`, `json.dumps`, `asdict` |
| [`RolloutBuffer.to_sft_dataset`](../saddlellm/RLScaling.py#L326)<br><sub>`to_sft_dataset(self) -> List[Dict[str, str]]`</sub> | method | `RolloutBuffer` 中实现数据集的公开操作。 | `sample.to_sft_record`, `self.accepted` |
| [`RLScalingTrainer.__init__`](../saddlellm/RLScaling.py#L333)<br><sub>`__init__(self, model=None, tokenizer=None, reward_manager: Optional[RewardManager]=None, rollout_fn: Optional[RolloutFn]=None, config: Optional[RLScalingConfig]=None, triage_assigner: Optional[TriageCreditAssigner]=None)`</sub> | method | 初始化 `RLScalingTrainer` 实例及其运行依赖。 | `RLScalingConfig`, `RolloutBuffer`, `random.seed` |
| [`RLScalingTrainer.collect`](../saddlellm/RLScaling.py#L351)<br><sub>`collect(self, prompts: Sequence[str], reward_fn: Optional[RewardFn]=None) -> RolloutBuffer`</sub> | method | `RLScalingTrainer` 中收集`collect`的公开操作。 | `ValueError`, `expanded_prompts.extend`, `self._rollout`, `RolloutSample`, `float`, `rewarder`, `zip`, `TriageCreditAssigner`, `TriageConfig`, `self.triage_assigner.assign` |
| [`RLScalingTrainer.train_grpo`](../saddlellm/RLScaling.py#L388)<br><sub>`train_grpo(self, ref_model, reward_fn: Optional[RewardFn]=None, num_steps: int=100, grpo_config=None)`</sub> | method | Run existing SaddleLLM GRPO on accepted prompts. | `ValueError`, `self.buffer.accepted`, `GRPOConfig`, `sorted`, `GRPOTrainer`, `trainer.train` |
| [`RLScalingTrainer.export_sft_jsonl`](../saddlellm/RLScaling.py#L423)<br><sub>`export_sft_jsonl(self, path: str, top_k_per_prompt: Optional[int]=None) -> str`</sub> | method | `RLScalingTrainer` 中导出`export_sft_jsonl`的公开操作。 | `self.buffer.accepted`, `values`, `self.buffer.by_prompt`, `selected.extend`, `sorted`, `os.makedirs`, `os.path.dirname`, `open`, `f.write`, `json.dumps` |
| [`RLScalingTrainer._rollout`](../saddlellm/RLScaling.py#L440)<br><sub>`_rollout(self, prompts: List[str]) -> List[str]`</sub> | method | `RLScalingTrainer` 中实现`rollout`的内部辅助逻辑。 | `self.rollout_fn`, `ValueError`, `next`, `self.model.parameters`, `self.model.eval`, `to`, `self.tokenizer`, `torch.no_grad`, `self.model.generate`, `self.tokenizer.decode` |
| [`RLScalingTrainer._annotate_group_advantages`](../saddlellm/RLScaling.py#L472)<br><sub>`_annotate_group_advantages(self, samples: List[RolloutSample])`</sub> | method | `RLScalingTrainer` 中实现`annotate_group_advantages`的内部辅助逻辑。 | `append`, `grouped.setdefault`, `grouped.values`, `sum`, `max`, `len`, `math.sqrt` |
| [`RLScalingTrainer._filter_samples`](../saddlellm/RLScaling.py#L487)<br><sub>`_filter_samples(self, samples: List[RolloutSample])`</sub> | method | `RLScalingTrainer` 中过滤`filter_samples`的内部辅助逻辑。 | `append`, `grouped.setdefault`, `grouped.values`, `metadata.get`, `sorted`, `set`, `id`, `len`, `_has_bad_repetition` |
| [`_has_bad_repetition`](../saddlellm/RLScaling.py#L519)<br><sub>`_has_bad_repetition(text: str) -> bool`</sub> | function | 模块级实现`has_bad_repetition`的内部辅助逻辑。 | `len`, `re.search`, `re.findall`, `text.lower`, `tuple`, `range`, `set` |

## `saddlellm/RapidDistill.py`

共 22 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelDownloader.download`](../saddlellm/RapidDistill.py#L240)<br><sub>`download(name: str, cache_dir: str=None)`</sub> | method | 下载一个小模型教师。 | `join`, `DOWNLOADABLE_TEACHERS.keys`, `KeyError`, `print`, `AutoModelForCausalLM.from_pretrained`, `torch.cuda.is_available`, `AutoTokenizer.from_pretrained` |
| [`ModelDownloader.list_available`](../saddlellm/RapidDistill.py#L266)<br><sub>`list_available()`</sub> | method | 列出所有可下载的教师。 | `print`, `DOWNLOADABLE_TEACHERS.items` |
| [`ModelDownloader.recommend_for`](../saddlellm/RapidDistill.py#L277)<br><sub>`recommend_for(student_params: int=None, student_size: str=None)`</sub> | method | 根据学生模型大小推荐最佳教师。 | `size_map.get`, `student_size.lower`, `DOWNLOADABLE_TEACHERS.items`, `recommendations.append`, `recommendations.sort`, `float`, `replace`, `params.replace` |
| [`LocalTeacherPool.__init__`](../saddlellm/RapidDistill.py#L310)<br><sub>`__init__(self, cache_dir: str=None)`</sub> | method | 初始化 `LocalTeacherPool` 实例及其运行依赖。 | — |
| [`LocalTeacherPool.add`](../saddlellm/RapidDistill.py#L314)<br><sub>`add(self, name: str)`</sub> | method | 下载并添加一个教师。 | `ModelDownloader.download` |
| [`LocalTeacherPool.add_all`](../saddlellm/RapidDistill.py#L320)<br><sub>`add_all(self, names: List[str]=None)`</sub> | method | 批量添加教师。 | `list`, `DOWNLOADABLE_TEACHERS.keys`, `self.add`, `print` |
| [`LocalTeacherPool.generate`](../saddlellm/RapidDistill.py#L330)<br><sub>`generate(self, teacher_name: str, prompt: str, max_tokens: int=1024) -> str`</sub> | method | 用指定教师生成回复。 | `model.eval`, `next`, `model.parameters`, `to`, `tokenizer`, `torch.no_grad`, `model.generate`, `tokenizer.decode`, `len`, `strip` |
| [`LocalTeacherPool.batch_generate`](../saddlellm/RapidDistill.py#L347)<br><sub>`batch_generate(self, teacher_name: str, prompts: List[str], max_tokens: int=1024) -> List[str]`</sub> | method | `LocalTeacherPool` 中生成批次的公开操作。 | `self.generate` |
| [`LocalTeacherPool.remove`](../saddlellm/RapidDistill.py#L350)<br><sub>`remove(self, name: str)`</sub> | method | 释放模型显存。 | `torch.cuda.is_available`, `torch.cuda.empty_cache` |
| [`LocalTeacherPool.list_loaded`](../saddlellm/RapidDistill.py#L356)<br><sub>`list_loaded(self)`</sub> | method | `LocalTeacherPool` 中列出`list_loaded`的公开操作。 | `print`, `len`, `self.models.items` |
| [`quick_local_distill`](../saddlellm/RapidDistill.py#L366)<br><sub>`quick_local_distill(student_model, student_tokenizer, teacher_names: List[str]=None, prompts: List[str]=None, total_examples: int=300, output_dir: str='./local_distilled_model', auto_cleanup: bool=True)`</sub> | function | 快速本地蒸馏 — 下载小模型教师 → 蒸 → 释放显存。 | `sum`, `p.numel`, `student_model.parameters`, `print`, `ModelDownloader.recommend_for`, `LocalTeacherPool`, `pool.add_all`, `RuntimeError`, `RapidDistill._generate_diverse_prompts`, `len` |
| [`SmartRouter.route`](../saddlellm/RapidDistill.py#L510)<br><sub>`route(self, prompt: str, available_teachers: List[str]=None) -> str`</sub> | method | 为 prompt 选择最佳教师。 | `list`, `TEACHER_PROFILES.keys`, `prompt.lower`, `any`, `next`, `len`, `re.findall` |
| [`ParallelAPICaller.__init__`](../saddlellm/RapidDistill.py#L551)<br><sub>`__init__(self, api_keys: Dict[str, str]=None)`</sub> | method | 初始化 `ParallelAPICaller` 实例及其运行依赖。 | `threading.Lock` |
| [`ParallelAPICaller._get_teacher`](../saddlellm/RapidDistill.py#L557)<br><sub>`_get_teacher(self, name: str)`</sub> | method | `ParallelAPICaller` 中读取`get_teacher`的内部辅助逻辑。 | `self.api_keys.get`, `os.environ.get`, `profile.provider.upper`, `TeacherInterface.from_openai`, `TeacherInterface.from_anthropic`, `TeacherInterface.from_deepseek`, `TeacherInterface` |
| [`ParallelAPICaller.batch_call`](../saddlellm/RapidDistill.py#L580)<br><sub>`batch_call(self, prompts: List[str], teacher_names: List[str]=None, router: 'SmartRouter'=None, concurrency: int=10, on_progress: Callable=None) -> List[Dict]`</sub> | method | 并行调用多个教师 API，为每个 prompt 选择最佳教师。 | `SmartRouter`, `self._get_teacher`, `ThreadPoolExecutor`, `executor.submit`, `enumerate`, `as_completed`, `future.result`, `results.append`, `on_progress`, `len` |
| [`ParallelAPICaller.batch_call.process_one`](../saddlellm/RapidDistill.py#L603)<br><sub>`process_one(i, prompt)`</sub> | nested function | 处理单个 prompt。 | `router.route`, `self._get_teacher`, `teacher.generate`, `len`, `round`, `logger.warning`, `fb.generate`, `str` |
| [`ParallelAPICaller.get_cost_summary`](../saddlellm/RapidDistill.py#L651)<br><sub>`get_cost_summary(self) -> Dict`</sub> | method | `ParallelAPICaller` 中读取`get_cost_summary`的公开操作。 | `dict` |
| [`RapidDistill.run`](../saddlellm/RapidDistill.py#L688)<br><sub>`run(student_model, student_tokenizer, prompts: List[str]=None, total_examples: int=500, use_teachers: List[str]=None, mode: str='smart_route', api_keys: Dict[str, str]=None, output_dir: str='./rapid_distilled_model', domains: Dict[str, List[str]]=None, concurrency: int=10) -> Dict`</sub> | method | 一键快速蒸馏。 | `SmartRouter`, `ParallelAPICaller`, `logger.info`, `RapidDistill._generate_diverse_prompts`, `len`, `caller.batch_call`, `all_results.append`, `range`, `max`, `RapidDistill._quality_heuristic` |
| [`RapidDistill._generate_diverse_prompts`](../saddlellm/RapidDistill.py#L806)<br><sub>`_generate_diverse_prompts(total: int, domains: Dict[str, List[str]]=None) -> List[str]`</sub> | method | 生成多样化 prompts。 | `domains.values`, `all_prompts.extend`, `random.shuffle`, `list`, `templates.keys`, `len`, `random.choice`, `prompts.append`, `template.format` |
| [`RapidDistill._quality_heuristic`](../saddlellm/RapidDistill.py#L850)<br><sub>`_quality_heuristic(text: str) -> float`</sub> | method | `RapidDistill` 中实现`quality_heuristic`的内部辅助逻辑。 | `len`, `any`, `min` |
| [`RapidDistill._count_teachers`](../saddlellm/RapidDistill.py#L861)<br><sub>`_count_teachers(results: List[Dict]) -> Dict`</sub> | method | `RapidDistill` 中实现`count_teachers`的内部辅助逻辑。 | `r.get`, `counts.get` |
| [`RapidDistill.estimate_cost`](../saddlellm/RapidDistill.py#L869)<br><sub>`estimate_cost(prompts: List[str], teachers: List[str]=None) -> Dict`</sub> | method | 预估蒸馏费用。 | `sum`, `len`, `round`, `int` |

## `saddlellm/RealWorldOperators.py`

共 39 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`_value`](../saddlellm/RealWorldOperators.py#L34)<br><sub>`_value(config: Mapping[str, Any], *keys: str, default: Any=None) -> Any`</sub> | function | 模块级实现`value`的内部辅助逻辑。 | — |
| [`_as_bool`](../saddlellm/RealWorldOperators.py#L41)<br><sub>`_as_bool(value: Any, default: bool=False) -> bool`</sub> | function | 模块级实现`as_bool`的内部辅助逻辑。 | `isinstance`, `lower`, `strip`, `str` |
| [`_as_list`](../saddlellm/RealWorldOperators.py#L49)<br><sub>`_as_list(value: Any, default: Sequence[str]=()) -> List[str]`</sub> | function | 模块级列出`as_list`的内部辅助逻辑。 | `list`, `isinstance`, `part.strip`, `value.split`, `strip`, `str` |
| [`_json_safe`](../saddlellm/RealWorldOperators.py#L57)<br><sub>`_json_safe(value: Any) -> Any`</sub> | function | 模块级实现`json_safe`的内部辅助逻辑。 | `isinstance`, `math.isfinite`, `str`, `_json_safe`, `value.items`, `hasattr`, `value.item` |
| [`_write_json`](../saddlellm/RealWorldOperators.py#L76)<br><sub>`_write_json(path: Path, payload: Mapping[str, Any]) -> str`</sub> | function | 模块级写入`write_json`的内部辅助逻辑。 | `path.parent.mkdir`, `path.open`, `json.dump`, `_json_safe`, `str`, `path.resolve` |
| [`_artifact_bytes`](../saddlellm/RealWorldOperators.py#L83)<br><sub>`_artifact_bytes(path: Path) -> int`</sub> | function | 模块级实现产物的内部辅助逻辑。 | `path.exists`, `path.is_file`, `path.stat`, `sum`, `item.stat`, `path.rglob`, `item.is_file` |
| [`_resolve_path`](../saddlellm/RealWorldOperators.py#L91)<br><sub>`_resolve_path(raw_path: Any, work_dir: Path) -> Path`</sub> | function | 模块级解析路径的内部辅助逻辑。 | `ValueError`, `Path`, `os.path.expandvars`, `os.path.expanduser`, `str`, `raw.is_absolute`, `candidates.extend`, `Path.cwd`, `candidate.exists`, `candidate.resolve` |
| [`_parse_hf_uri`](../saddlellm/RealWorldOperators.py#L104)<br><sub>`_parse_hf_uri(uri: str) -> Tuple[str, str, str]`</sub> | function | Parse ``hf://owner/dataset/config/split`` without losing the owner. | `split`, `len`, `ValueError`, `join` |
| [`_cached_hf_parquet_files`](../saddlellm/RealWorldOperators.py#L119)<br><sub>`_cached_hf_parquet_files(dataset_id: str, config_name: str, split: str) -> List[Path]`</sub> | function | 模块级实现`cached_hf_parquet_files`的内部辅助逻辑。 | `Path`, `os.getenv`, `os.path.join`, `str`, `Path.home`, `dataset_id.replace`, `dataset_id.split`, `dict.fromkeys`, `repository.exists`, `main_ref.exists` |
| [`_record_text`](../saddlellm/RealWorldOperators.py#L148)<br><sub>`_record_text(record: Mapping[str, Any], text_column: str) -> str`</sub> | function | 模块级记录`record_text`的内部辅助逻辑。 | `record.get`, `strip`, `str`, `parts.append`, `join` |
| [`_load_texts`](../saddlellm/RealWorldOperators.py#L169)<br><sub>`_load_texts(path: Path, text_column: str='text', limit: int=0) -> List[str]`</sub> | function | 模块级加载`load_texts`的内部辅助逻辑。 | `path.suffix.lower`, `path.open`, `line.strip`, `loaded.append`, `json.loads`, `len`, `json.load`, `isinstance`, `payload.get`, `pd.read_parquet` |
| [`_load_text_source`](../saddlellm/RealWorldOperators.py#L215)<br><sub>`_load_text_source(raw_source: Any, work_dir: Path, text_column: str, limit: int) -> Tuple[List[str], str]`</sub> | function | 模块级加载数据源的内部辅助逻辑。 | `strip`, `str`, `source.startswith`, `_resolve_path`, `_load_texts`, `_parse_hf_uri`, `_cached_hf_parquet_files`, `pd.read_parquet`, `frames.append`, `to_dict` |
| [`_seed_everything`](../saddlellm/RealWorldOperators.py#L254)<br><sub>`_seed_everything(seed: int) -> None`</sub> | function | 模块级实现`seed_everything`的内部辅助逻辑。 | `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.is_available`, `torch.cuda.manual_seed_all` |
| [`_device`](../saddlellm/RealWorldOperators.py#L269)<br><sub>`_device(config: Mapping[str, Any]) -> str`</sub> | function | 模块级实现设备的内部辅助逻辑。 | `lower`, `str`, `_value`, `torch.cuda.is_available`, `requested.startswith`, `RuntimeError` |
| [`_model_ref`](../saddlellm/RealWorldOperators.py#L280)<br><sub>`_model_ref(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]]) -> str`</sub> | function | 模块级实现模型的内部辅助逻辑。 | `upstream.get`, `isinstance`, `artifact.get`, `str`, `_value` |
| [`_load_tokenizer`](../saddlellm/RealWorldOperators.py#L291)<br><sub>`_load_tokenizer(model_ref: str, config: Mapping[str, Any])`</sub> | function | 模块级加载分词器的内部辅助逻辑。 | `str`, `_value`, `_as_bool`, `AutoTokenizer.from_pretrained`, `tokenizer.add_special_tokens` |
| [`_load_model`](../saddlellm/RealWorldOperators.py#L311)<br><sub>`_load_model(model_ref: str, config: Mapping[str, Any], device: str, for_training: bool=False)`</sub> | function | 模块级加载模型的内部辅助逻辑。 | `lower`, `str`, `_value`, `device.startswith`, `dtype_map.get`, `_as_bool`, `AutoModelForCausalLM.from_pretrained`, `model.to` |
| [`_token_batch`](../saddlellm/RealWorldOperators.py#L344)<br><sub>`_token_batch(tokenizer, texts: Sequence[str], max_length: int, device: str)`</sub> | function | 模块级实现批次的内部辅助逻辑。 | `tokenizer`, `list`, `value.to`, `encoded.items`, `clone` |
| [`_batches`](../saddlellm/RealWorldOperators.py#L359)<br><sub>`_batches(texts: Sequence[str], batch_size: int, steps: int=0, seed: int=42) -> Iterable[List[str]]`</sub> | function | 模块级实现`batches`的内部辅助逻辑。 | `list`, `range`, `len`, `random.Random`, `rng.shuffle` |
| [`_autocast`](../saddlellm/RealWorldOperators.py#L376)<br><sub>`_autocast(device: str)`</sub> | function | 模块级实现`autocast`的内部辅助逻辑。 | `device.startswith`, `torch.autocast`, `contextlib.nullcontext` |
| [`_evaluate_torch`](../saddlellm/RealWorldOperators.py#L384)<br><sub>`_evaluate_torch(model, tokenizer, texts: Sequence[str], max_length: int, batch_size: int, device: str) -> Dict[str, float]`</sub> | function | 模块级评估`evaluate_torch`的内部辅助逻辑。 | `model.eval`, `time.perf_counter`, `torch.inference_mode`, `_batches`, `_token_batch`, `_autocast`, `model`, `int`, `item`, `sum` |
| [`_parameter_metrics`](../saddlellm/RealWorldOperators.py#L408)<br><sub>`_parameter_metrics(model) -> Dict[str, float]`</sub> | function | 模块级实现指标的内部辅助逻辑。 | `sum`, `parameter.numel`, `model.parameters`, `parameter.element_size`, `int`, `float` |
| [`_normal_result`](../saddlellm/RealWorldOperators.py#L420)<br><sub>`_normal_result(operator: str, output_dir: Path, message: str, artifact: Mapping[str, Any], metrics: Mapping[str, Any], lineage: Optional[Mapping[str, Any]]=None, warnings: Optional[Sequence[str]]=None) -> Dict[str, Any]`</sub> | function | 模块级实现`normal_result`的内部辅助逻辑。 | `dict`, `list`, `artifact.get`, `str`, `manifest_path.resolve`, `_write_json`, `_json_safe` |
| [`_load_operator`](../saddlellm/RealWorldOperators.py#L448)<br><sub>`_load_operator(config: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]`</sub> | function | 模块级加载算子的内部辅助逻辑。 | `time.perf_counter`, `_device`, `_model_ref`, `_load_tokenizer`, `_load_model`, `_parameter_metrics`, `metrics.update`, `int`, `len`, `str` |
| [`_data_config`](../saddlellm/RealWorldOperators.py#L479)<br><sub>`_data_config(config: Mapping[str, Any], output_dir: Path) -> Tuple[List[str], List[str], Dict[str, Any]]`</sub> | function | 模块级实现数据、配置的内部辅助逻辑。 | `str`, `_value`, `int`, `_load_text_source`, `min`, `max`, `len` |
| [`_sft_operator`](../saddlellm/RealWorldOperators.py#L503)<br><sub>`_sft_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]`</sub> | function | 模块级实现算子的内部辅助逻辑。 | `int`, `_value`, `_seed_everything`, `_device`, `_model_ref`, `_data_config`, `max`, `float`, `_load_tokenizer`, `_load_model` |
| [`_copy_sliced_teacher`](../saddlellm/RealWorldOperators.py#L593)<br><sub>`_copy_sliced_teacher(teacher, student) -> int`</sub> | function | Initialise a narrower GPT-2 student with deterministic teacher slices. | `teacher.state_dict`, `student.state_dict`, `int`, `getattr`, `round`, `range`, `no_grad`, `__import__`, `student_state.items`, `name.startswith` |
| [`_distillation_operator`](../saddlellm/RealWorldOperators.py#L627)<br><sub>`_distillation_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]`</sub> | function | 模块级实现算子的内部辅助逻辑。 | `int`, `_value`, `_seed_everything`, `_device`, `_model_ref`, `_data_config`, `float`, `_load_tokenizer`, `_load_model`, `teacher.eval` |
| [`_prunable_parameters`](../saddlellm/RealWorldOperators.py#L753)<br><sub>`_prunable_parameters(model, targets: Sequence[str])`</sub> | function | 模块级实现`prunable_parameters`的内部辅助逻辑。 | `model.named_parameters`, `parameter.is_floating_point`, `any`, `selected.append` |
| [`_zero_sparsity`](../saddlellm/RealWorldOperators.py#L764)<br><sub>`_zero_sparsity(parameters: Sequence[Tuple[str, Any]]) -> Tuple[int, int]`</sub> | function | 模块级实现`zero_sparsity`的内部辅助逻辑。 | `sum`, `int`, `item`, `parameter.numel` |
| [`_global_magnitude_threshold`](../saddlellm/RealWorldOperators.py#L770)<br><sub>`_global_magnitude_threshold(parameters: Sequence[Tuple[str, Any]], sparsity: float, bins: int=32768) -> float`</sub> | function | Find a deterministic global threshold without concatenating huge tensors. | `max`, `float`, `item`, `abs`, `parameter.detach`, `torch.zeros`, `double`, `cpu`, `torch.histc`, `values.numel` |
| [`_pruning_operator`](../saddlellm/RealWorldOperators.py#L795)<br><sub>`_pruning_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]`</sub> | function | 模块级实现算子的内部辅助逻辑。 | `int`, `_value`, `_seed_everything`, `_device`, `_model_ref`, `_data_config`, `float`, `ValueError`, `_load_tokenizer`, `_load_model` |
| [`_export_onnx`](../saddlellm/RealWorldOperators.py#L862)<br><sub>`_export_onnx(model, tokenizer, output_path: Path, max_length: int, opset: int) -> None`</sub> | function | 模块级导出`export_onnx`的内部辅助逻辑。 | `eval`, `float`, `model.to`, `LogitsOnly`, `tokenizer`, `output_path.parent.mkdir`, `torch.inference_mode`, `torch.onnx.export`, `str` |
| [`_export_onnx.LogitsOnly.__init__`](../saddlellm/RealWorldOperators.py#L866)<br><sub>`__init__(self, wrapped)`</sub> | nested function | 初始化 `LogitsOnly` 实例及其运行依赖。 | `__init__`, `super` |
| [`_export_onnx.LogitsOnly.forward`](../saddlellm/RealWorldOperators.py#L870)<br><sub>`forward(self, input_ids, attention_mask)`</sub> | nested function | 执行 `LogitsOnly` 的前向计算。 | `self.wrapped` |
| [`_evaluate_onnx`](../saddlellm/RealWorldOperators.py#L900)<br><sub>`_evaluate_onnx(model_path: Path, tokenizer, texts: Sequence[str], max_length: int, warmup_runs: int, benchmark_runs: int) -> Dict[str, float]`</sub> | function | 模块级评估`evaluate_onnx`的内部辅助逻辑。 | `ort.InferenceSession`, `str`, `tokenizer`, `list`, `astype`, `range`, `max`, `session.run`, `time.perf_counter`, `durations.append` |
| [`_quantization_operator`](../saddlellm/RealWorldOperators.py#L950)<br><sub>`_quantization_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]`</sub> | function | 模块级实现算子的内部辅助逻辑。 | `_model_ref`, `_data_config`, `int`, `_value`, `_as_bool`, `_as_list`, `_load_tokenizer`, `dict`, `_load_model`, `time.perf_counter` |
| [`_evaluation_operator`](../saddlellm/RealWorldOperators.py#L1043)<br><sub>`_evaluation_operator(config: Mapping[str, Any], upstream: Optional[Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]`</sub> | function | 模块级实现算子的内部辅助逻辑。 | `ValueError`, `_data_config`, `int`, `_value`, `isinstance`, `upstream.get`, `str`, `artifact.get`, `_model_ref`, `_load_tokenizer` |
| [`run_large_model_operator`](../saddlellm/RealWorldOperators.py#L1097)<br><sub>`run_large_model_operator(label: str, config: Optional[Mapping[str, Any]], upstream_result: Optional[Mapping[str, Any]], work_dir: str) -> Dict[str, Any]`</sub> | function | Execute one visual LLM operator and return a JSON-safe result contract. | `ValueError`, `dict`, `resolve`, `Path`, `output_dir.mkdir`, `_load_operator`, `_sft_operator`, `_distillation_operator`, `_pruning_operator`, `_quantization_operator` |

## `saddlellm/ReleaseGate.py`

共 8 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ReleaseGateResult.to_dict`](../saddlellm/ReleaseGate.py#L36)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `ReleaseGateResult` 转为可序列化字典。 | `asdict` |
| [`ReleaseGateResult.save`](../saddlellm/ReleaseGate.py#L39)<br><sub>`save(self, path: str) -> str`</sub> | method | `ReleaseGateResult` 中保存`save`的公开操作。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump`, `self.to_dict`, `handle.write` |
| [`EvaluationReleaseGate.validate_rules`](../saddlellm/ReleaseGate.py#L62)<br><sub>`validate_rules(cls, rules: Mapping[str, Any]) -> List[str]`</sub> | method | `EvaluationReleaseGate` 中校验`validate_rules`的公开操作。 | `isinstance`, `rules.items`, `strip`, `str`, `issues.append`, `raw_rule.get`, `float`, `math.isfinite` |
| [`EvaluationReleaseGate.evaluate`](../saddlellm/ReleaseGate.py#L97)<br><sub>`evaluate(cls, metrics: Mapping[str, Any], rules: Mapping[str, Any], *, enabled: bool=True, require_all: bool=True) -> ReleaseGateResult`</sub> | method | `EvaluationReleaseGate` 中评估`evaluate`的公开操作。 | `ReleaseGateResult`, `cls.validate_rules`, `rules.items`, `bool`, `raw_rule.get`, `cls._optional_float`, `cls._find_metric`, `str`, `cls._score`, `checks.append` |
| [`EvaluationReleaseGate._optional_float`](../saddlellm/ReleaseGate.py#L194)<br><sub>`_optional_float(value: Any) -> Optional[float]`</sub> | method | `EvaluationReleaseGate` 中实现`optional_float`的内部辅助逻辑。 | `float` |
| [`EvaluationReleaseGate._find_metric`](../saddlellm/ReleaseGate.py#L198)<br><sub>`_find_metric(metrics: Mapping[str, Any], path: str) -> Any`</sub> | method | `EvaluationReleaseGate` 中查找`find_metric`的内部辅助逻辑。 | `path.split`, `isinstance` |
| [`EvaluationReleaseGate._score`](../saddlellm/ReleaseGate.py#L209)<br><sub>`_score(value: Any) -> Optional[float]`</sub> | method | `EvaluationReleaseGate` 中评分`score`的内部辅助逻辑。 | `isinstance`, `value.get`, `float`, `math.isfinite` |
| [`evaluate_release_gate`](../saddlellm/ReleaseGate.py#L219)<br><sub>`evaluate_release_gate(metrics: Mapping[str, Any], rules: Mapping[str, Any], *, enabled: bool=True, require_all: bool=True) -> Dict[str, Any]`</sub> | function | 模块级评估发布包的公开操作。 | `to_dict`, `EvaluationReleaseGate.evaluate` |

## `saddlellm/SaddleModeling.py`

共 58 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SaddleModelConfig.__post_init__`](../saddlellm/SaddleModeling.py#L67)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `SaddleModelConfig` 创建后校验并规范化字段。 | `dict`, `len` |
| [`SaddleModelConfig._layer_blueprint_dict`](../saddlellm/SaddleModeling.py#L75)<br><sub>`_layer_blueprint_dict(layer) -> Dict[str, Any]`</sub> | method | `SaddleModelConfig` 中实现模型蓝图的内部辅助逻辑。 | `asdict` |
| [`SaddleModelConfig.from_blueprint`](../saddlellm/SaddleModeling.py#L84)<br><sub>`from_blueprint(cls, blueprint) -> 'SaddleModelConfig'`</sub> | method | `SaddleModelConfig` 中实现模型蓝图的公开操作。 | `blueprint.expanded_layers`, `cls`, `getattr`, `cls._layer_blueprint_dict`, `blueprint.to_config_dict` |
| [`SaddleModelConfig.from_model_spec`](../saddlellm/SaddleModeling.py#L126)<br><sub>`from_model_spec(cls, spec) -> 'SaddleModelConfig'`</sub> | method | `SaddleModelConfig` 中实现模型的公开操作。 | `cls` |
| [`SaddleModelConfig.to_dict`](../saddlellm/SaddleModeling.py#L157)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `SaddleModelConfig` 转为可序列化字典。 | `asdict` |
| [`SaddleModelConfig.for_layer`](../saddlellm/SaddleModeling.py#L160)<br><sub>`for_layer(self, index: int) -> 'SaddleModelConfig'`</sub> | method | Return a standalone config resolved for one decoder layer. | `replace`, `len`, `IndexError`, `dict`, `layer.get`, `str`, `attention.get`, `ffn.get`, `residual.get` |
| [`SaddleModelConfig.save_pretrained`](../saddlellm/SaddleModeling.py#L221)<br><sub>`save_pretrained(self, path: str) -> str`</sub> | method | 保存 `SaddleModelConfig`，遵循预训练模型的目录契约。 | `os.makedirs`, `open`, `os.path.join`, `json.dump`, `self.to_dict` |
| [`SaddleRMSNorm.__init__`](../saddlellm/SaddleModeling.py#L248)<br><sub>`__init__(self, hidden_size: int, eps: float=1e-06)`</sub> | method | 初始化 `SaddleRMSNorm` 实例及其运行依赖。 | `__init__`, `super`, `nn.Parameter`, `torch.ones` |
| [`SaddleRMSNorm.forward`](../saddlellm/SaddleModeling.py#L253)<br><sub>`forward(self, x: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `SaddleRMSNorm` 的前向计算。 | `mean`, `x.pow`, `torch.rsqrt` |
| [`SaddleRotaryEmbedding.__init__`](../saddlellm/SaddleModeling.py#L259)<br><sub>`__init__(self, dim: int, max_position_embeddings: int=4096, base: float=10000.0, scaling_factor: float=1.0)`</sub> | method | 初始化 `SaddleRotaryEmbedding` 实例及其运行依赖。 | `__init__`, `super`, `float`, `torch.arange`, `self.register_buffer` |
| [`SaddleRotaryEmbedding.forward`](../saddlellm/SaddleModeling.py#L272)<br><sub>`forward(self, seq_len: int, device=None, dtype=None, offset: int=0) -> Tuple[torch.Tensor, torch.Tensor]`</sub> | method | 执行 `SaddleRotaryEmbedding` 的前向计算。 | `torch.arange`, `torch.outer`, `self.inv_freq.to`, `torch.cat`, `to`, `emb.cos`, `emb.sin` |
| [`_rotate_half`](../saddlellm/SaddleModeling.py#L283)<br><sub>`_rotate_half(x: torch.Tensor) -> torch.Tensor`</sub> | function | 模块级实现`rotate_half`的内部辅助逻辑。 | `torch.cat` |
| [`apply_rotary_pos_emb`](../saddlellm/SaddleModeling.py#L289)<br><sub>`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor)`</sub> | function | 模块级应用`apply_rotary_pos_emb`的公开操作。 | `_rotate_half` |
| [`apply_rotary_single`](../saddlellm/SaddleModeling.py#L295)<br><sub>`apply_rotary_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor`</sub> | function | 模块级应用`apply_rotary_single`的公开操作。 | `_rotate_half` |
| [`SaddleAttention.__init__`](../saddlellm/SaddleModeling.py#L302)<br><sub>`__init__(self, config: SaddleModelConfig)`</sub> | method | 初始化 `SaddleAttention` 实例及其运行依赖。 | `__init__`, `super`, `ValueError`, `nn.Linear`, `max`, `SaddleRotaryEmbedding`, `get` |
| [`SaddleAttention.forward`](../saddlellm/SaddleModeling.py#L335)<br><sub>`forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Any]=None, use_cache: bool=False) -> Tuple[torch.Tensor, Optional[Any]]`</sub> | method | 执行 `SaddleAttention` 的前向计算。 | `self.q_b_proj`, `self.q_a_proj`, `self.kv_a_proj`, `isinstance`, `past_key_value.get`, `torch.cat`, `transpose`, `q.view`, `view`, `self.k_b_proj` |
| [`SaddleAttention._attention`](../saddlellm/SaddleModeling.py#L409)<br><sub>`_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor`</sub> | method | `SaddleAttention` 中实现注意力的内部辅助逻辑。 | `AttentionBackendRegistry.run` |
| [`SaddleAttention._build_attention_mask`](../saddlellm/SaddleModeling.py#L429)<br><sub>`_build_attention_mask(attention_mask: Optional[torch.Tensor], q_len: int, kv_len: int, past_len: int, dtype: torch.dtype, device: torch.device, sliding_window: int=0) -> Tuple[Optional[torch.Tensor], bool]`</sub> | method | `SaddleAttention` 中构建注意力的内部辅助逻辑。 | `torch.finfo`, `torch.arange`, `torch.zeros`, `causal.masked_fill`, `int`, `torch.ones`, `torch.cat`, `to` |
| [`SaddleSwiGLU.__init__`](../saddlellm/SaddleModeling.py#L467)<br><sub>`__init__(self, hidden_size: int, intermediate_size: int)`</sub> | method | 初始化 `SaddleSwiGLU` 实例及其运行依赖。 | `__init__`, `super`, `nn.Linear` |
| [`SaddleSwiGLU.forward`](../saddlellm/SaddleModeling.py#L473)<br><sub>`forward(self, x: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `SaddleSwiGLU` 的前向计算。 | `self.down_proj`, `F.silu`, `self.gate_proj`, `self.up_proj` |
| [`SaddleMoE.__init__`](../saddlellm/SaddleModeling.py#L478)<br><sub>`__init__(self, config: SaddleModelConfig)`</sub> | method | 初始化 `SaddleMoE` 实例及其运行依赖。 | `__init__`, `super`, `nn.Linear`, `nn.ModuleList`, `SaddleSwiGLU`, `range`, `self.register_buffer`, `torch.zeros` |
| [`SaddleMoE.forward`](../saddlellm/SaddleModeling.py#L494)<br><sub>`forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]`</sub> | method | 执行 `SaddleMoE` 的前向计算。 | `hidden_states.reshape`, `self.router`, `torch.topk`, `F.softmax`, `torch.zeros_like`, `flat.new_zeros`, `enumerate`, `matches.any`, `token_mask.any`, `self._zero_parameter_connection` |
| [`SaddleMoE._global_expert_load`](../saddlellm/SaddleModeling.py#L550)<br><sub>`_global_expert_load(self, load: torch.Tensor) -> torch.Tensor`</sub> | method | Sum routing counts across DDP ranks when a process group is active. | `clone`, `self.last_expert_load.detach`, `distributed.is_available`, `distributed.is_initialized`, `load.detach`, `distributed.all_reduce` |
| [`SaddleMoE._global_router_probabilities`](../saddlellm/SaddleModeling.py#L566)<br><sub>`_global_router_probabilities(probabilities: torch.Tensor) -> torch.Tensor`</sub> | method | Average differentiable router probabilities across active ranks. | `distributed.is_available`, `distributed.is_initialized`, `distributed.get_world_size`, `clone`, `probabilities.detach`, `distributed.all_reduce`, `global_value.div_` |
| [`SaddleMoE._zero_parameter_connection`](../saddlellm/SaddleModeling.py#L583)<br><sub>`_zero_parameter_connection(module: nn.Module, reference: torch.Tensor) -> torch.Tensor`</sub> | method | `SaddleMoE` 中实现`zero_parameter_connection`的内部辅助逻辑。 | `reference.new_zeros`, `module.parameters`, `parameter.numel`, `parameter.reshape` |
| [`SaddleMoE._router_aux_loss`](../saddlellm/SaddleModeling.py#L590)<br><sub>`_router_aux_loss(self, logits: torch.Tensor, load: torch.Tensor) -> torch.Tensor`</sub> | method | `SaddleMoE` 中实现`router_aux_loss`的内部辅助逻辑。 | `mean`, `F.softmax`, `self._global_router_probabilities`, `load.float`, `clamp_min`, `sum`, `torch.sum` |
| [`SaddleMoE._update_router_bias`](../saddlellm/SaddleModeling.py#L598)<br><sub>`_update_router_bias(self, load: torch.Tensor, rate: float=0.001)`</sub> | method | `SaddleMoE` 中更新`update_router_bias`的内部辅助逻辑。 | `torch.no_grad`, `clamp_min`, `load.mean` |
| [`SaddleMoE._router_entropy`](../saddlellm/SaddleModeling.py#L603)<br><sub>`_router_entropy(self, logits: torch.Tensor) -> torch.Tensor`</sub> | method | `SaddleMoE` 中实现`router_entropy`的内部辅助逻辑。 | `F.softmax`, `mean`, `sum`, `torch.log`, `probs.clamp_min`, `math.log`, `max` |
| [`SaddleMoE._load_balance`](../saddlellm/SaddleModeling.py#L608)<br><sub>`_load_balance(self, load: torch.Tensor) -> torch.Tensor`</sub> | method | `SaddleMoE` 中加载`load_balance`的内部辅助逻辑。 | `clamp_min`, `load.sum`, `sum`, `torch.log`, `probs.clamp_min`, `math.log`, `max` |
| [`SaddleDecoderLayer.__init__`](../saddlellm/SaddleModeling.py#L615)<br><sub>`__init__(self, config: SaddleModelConfig, layer_index: Optional[int]=None, total_layers: Optional[int]=None)`</sub> | method | 初始化 `SaddleDecoderLayer` 实例及其运行依赖。 | `__init__`, `super`, `SaddleRMSNorm`, `SaddleAttention`, `SaddleMoE`, `SaddleSwiGLU`, `nn.Parameter`, `torch.tensor`, `float`, `nn.Dropout` |
| [`SaddleDecoderLayer.forward`](../saddlellm/SaddleModeling.py#L648)<br><sub>`forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache: bool=False)`</sub> | method | 执行 `SaddleDecoderLayer` 的前向计算。 | `self.self_attn`, `self.input_layernorm`, `self.residual_dropout`, `self.post_attention_layernorm`, `self.mlp`, `residual.new_zeros` |
| [`SaddleForCausalLM.__init__`](../saddlellm/SaddleModeling.py#L685)<br><sub>`__init__(self, config: SaddleModelConfig)`</sub> | method | 初始化 `SaddleForCausalLM` 实例及其运行依赖。 | `__init__`, `super`, `nn.Embedding`, `config.for_layer`, `range`, `nn.ModuleList`, `SaddleDecoderLayer`, `enumerate`, `SaddleRMSNorm`, `nn.Linear` |
| [`SaddleForCausalLM.from_blueprint`](../saddlellm/SaddleModeling.py#L717)<br><sub>`from_blueprint(cls, blueprint) -> 'SaddleForCausalLM'`</sub> | method | `SaddleForCausalLM` 中实现模型蓝图的公开操作。 | `cls`, `SaddleModelConfig.from_blueprint` |
| [`SaddleForCausalLM.from_model_spec`](../saddlellm/SaddleModeling.py#L721)<br><sub>`from_model_spec(cls, spec) -> 'SaddleForCausalLM'`</sub> | method | `SaddleForCausalLM` 中实现模型的公开操作。 | `cls`, `SaddleModelConfig.from_model_spec` |
| [`SaddleForCausalLM.post_init`](../saddlellm/SaddleModeling.py#L724)<br><sub>`post_init(self)`</sub> | method | `SaddleForCausalLM` 中实现`post_init`的公开操作。 | `self.named_modules`, `isinstance`, `nn.init.normal_`, `math.sqrt`, `max`, `len`, `torch.no_grad`, `ValueError`, `self._residual_output_projections`, `projection.weight.zero_` |
| [`SaddleForCausalLM._residual_output_projections`](../saddlellm/SaddleModeling.py#L746)<br><sub>`_residual_output_projections(layer: SaddleDecoderLayer) -> List[nn.Linear]`</sub> | method | `SaddleForCausalLM` 中实现输出的内部辅助逻辑。 | `projections.extend`, `projections.append` |
| [`SaddleForCausalLM.forward`](../saddlellm/SaddleModeling.py#L756)<br><sub>`forward(self, input_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None, use_cache: bool=False, return_dict: bool=True, **_)`</sub> | method | 执行 `SaddleForCausalLM` 的前向计算。 | `ValueError`, `self.embed_tokens`, `bool`, `len`, `zip`, `dict`, `self._moe_checkpoint_contexts`, `checkpoint`, `layer`, `aux_losses.append` |
| [`SaddleForCausalLM.forward.custom_forward`](../saddlellm/SaddleModeling.py#L798)<br><sub>`custom_forward(states: torch.Tensor, current_layer: SaddleDecoderLayer=layer, current_past_key_value: Any=past_key_value)`</sub> | nested function | `SaddleForCausalLM` 中实现`custom_forward`的局部回调/辅助逻辑。 | `current_layer` |
| [`SaddleForCausalLM._moe_checkpoint_contexts`](../saddlellm/SaddleModeling.py#L855)<br><sub>`_moe_checkpoint_contexts(moe: SaddleMoE)`</sub> | method | `SaddleForCausalLM` 中实现检查点的内部辅助逻辑。 | `clone`, `moe.router_bias.detach`, `nullcontext`, `SaddleForCausalLM._moe_checkpoint_recompute_context` |
| [`SaddleForCausalLM._moe_checkpoint_recompute_context`](../saddlellm/SaddleModeling.py#L866)<br><sub>`_moe_checkpoint_recompute_context(moe: SaddleMoE, router_bias_snapshot: torch.Tensor)`</sub> | method | `SaddleForCausalLM` 中实现检查点的内部辅助逻辑。 | `clone`, `moe.router_bias.detach`, `moe.last_expert_load.detach`, `torch.no_grad`, `moe.router_bias.copy_`, `moe.last_expert_load.copy_` |
| [`SaddleForCausalLM.gradient_checkpointing_enable`](../saddlellm/SaddleModeling.py#L884)<br><sub>`gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict]=None)`</sub> | method | `SaddleForCausalLM` 中实现`gradient_checkpointing_enable`的公开操作。 | `dict`, `kwargs.pop`, `ValueError` |
| [`SaddleForCausalLM.gradient_checkpointing_disable`](../saddlellm/SaddleModeling.py#L902)<br><sub>`gradient_checkpointing_disable(self)`</sub> | method | `SaddleForCausalLM` 中实现`gradient_checkpointing_disable`的公开操作。 | — |
| [`SaddleForCausalLM.enable_input_require_grads`](../saddlellm/SaddleModeling.py#L906)<br><sub>`enable_input_require_grads(self)`</sub> | method | `SaddleForCausalLM` 中实现输入的公开操作。 | `self.embed_tokens.register_forward_hook` |
| [`SaddleForCausalLM.enable_input_require_grads.make_inputs_require_grad`](../saddlellm/SaddleModeling.py#L907)<br><sub>`make_inputs_require_grad(_module, _input, output)`</sub> | nested function | `SaddleForCausalLM` 中实现`make_inputs_require_grad`的局部回调/辅助逻辑。 | `output.requires_grad_` |
| [`SaddleForCausalLM.get_input_embeddings`](../saddlellm/SaddleModeling.py#L912)<br><sub>`get_input_embeddings(self)`</sub> | method | `SaddleForCausalLM` 中读取输入的公开操作。 | — |
| [`SaddleForCausalLM.set_input_embeddings`](../saddlellm/SaddleModeling.py#L915)<br><sub>`set_input_embeddings(self, value)`</sub> | method | `SaddleForCausalLM` 中设置输入的公开操作。 | — |
| [`SaddleForCausalLM.get_output_embeddings`](../saddlellm/SaddleModeling.py#L918)<br><sub>`get_output_embeddings(self)`</sub> | method | `SaddleForCausalLM` 中读取输出的公开操作。 | — |
| [`SaddleForCausalLM.set_output_embeddings`](../saddlellm/SaddleModeling.py#L921)<br><sub>`set_output_embeddings(self, value)`</sub> | method | `SaddleForCausalLM` 中设置输出的公开操作。 | — |
| [`SaddleForCausalLM.resize_token_embeddings`](../saddlellm/SaddleModeling.py#L924)<br><sub>`resize_token_embeddings(self, new_num_tokens: int)`</sub> | method | `SaddleForCausalLM` 中实现`resize_token_embeddings`的公开操作。 | `int`, `nn.Embedding`, `nn.Linear`, `min`, `torch.no_grad`, `copy_` |
| [`SaddleForCausalLM.prepare_inputs_for_generation`](../saddlellm/SaddleModeling.py#L938)<br><sub>`prepare_inputs_for_generation(self, input_ids: torch.Tensor, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> Dict`</sub> | method | `SaddleForCausalLM` 中准备`prepare_inputs_for_generation`的公开操作。 | `kwargs.get` |
| [`SaddleForCausalLM.router_metrics`](../saddlellm/SaddleModeling.py#L955)<br><sub>`router_metrics(self) -> Dict[str, float]`</sub> | method | Aggregate latest MoE router metrics across layers. | `getattr`, `float`, `layer.mlp.last_expert_load.detach`, `loads.append`, `tolist`, `load.cpu`, `balances.append`, `cpu`, `layer.mlp._load_balance`, `len` |
| [`SaddleForCausalLM._causal_lm_loss`](../saddlellm/SaddleModeling.py#L974)<br><sub>`_causal_lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor`</sub> | method | `SaddleForCausalLM` 中实现`causal_lm_loss`的内部辅助逻辑。 | `contiguous`, `F.cross_entropy`, `shift_logits.view`, `shift_logits.size`, `shift_labels.view` |
| [`SaddleForCausalLM._mtp_loss`](../saddlellm/SaddleModeling.py#L979)<br><sub>`_mtp_loss(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]`</sub> | method | `SaddleForCausalLM` 中实现`mtp_loss`的内部辅助逻辑。 | `hidden_states.new_zeros`, `enumerate`, `labels.size`, `head`, `F.cross_entropy`, `logits.reshape`, `logits.size`, `target.reshape`, `min`, `len` |
| [`SaddleForCausalLM.generate`](../saddlellm/SaddleModeling.py#L995)<br><sub>`generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, max_new_tokens: int=32, temperature: float=1.0, do_sample: Optional[bool]=None, top_p: float=1.0, top_k: int=0, repetition_penalty: float=1.0, pad_token_id: Optional[int]=None, eos_token_id: Optional[Any]=None, min_new_tokens: int=0, use_cache: bool=True, return_dict_in_generate: bool=False, output_scores: bool=False, **generation_kwargs) -> torch.Tensor`</sub> | method | Generate with the standard greedy/sampling arguments used by callers. | `generation_kwargs.pop`, `NotImplementedError`, `join`, `sorted`, `TypeError`, `ValueError`, `self.eval`, `torch.ones_like`, `attention_mask.to`, `int` |
| [`SaddleForCausalLM.draft_mtp_tokens`](../saddlellm/SaddleModeling.py#L1129)<br><sub>`draft_mtp_tokens(self, input_ids: torch.Tensor, max_draft_tokens: Optional[int]=None) -> torch.Tensor`</sub> | method | Return greedy MTP draft tokens predicted from the final hidden state. | `self.eval`, `input_ids.new_empty`, `self`, `squeeze`, `head`, `torch.stack`, `torch.argmax` |
| [`SaddleForCausalLM.generate_mtp_speculative`](../saddlellm/SaddleModeling.py#L1143)<br><sub>`generate_mtp_speculative(self, input_ids: torch.Tensor, max_new_tokens: int=32, max_draft_tokens: Optional[int]=None) -> torch.Tensor`</sub> | method | Experimental full-forward MTP draft/verify decoding. | `self.eval`, `self`, `torch.argmax`, `torch.cat`, `self.draft_mtp_tokens`, `draft.numel`, `range`, `torch.equal` |
| [`SaddleForCausalLM.save_pretrained`](../saddlellm/SaddleModeling.py#L1180)<br><sub>`save_pretrained(self, path: str, state_dict: Optional[Dict]=None, metadata: Optional[Dict[str, Any]]=None, parallelism: Optional[Dict[str, Any]]=None, **_) -> str`</sub> | method | 保存 `SaddleForCausalLM`，遵循预训练模型的目录契约。 | `os.makedirs`, `self.state_dict`, `torch.save`, `os.path.join`, `self.config.save_pretrained`, `type`, `isoformat`, `datetime.now`, `sum`, `parameter.numel` |
| [`SaddleForCausalLM.from_pretrained_saddle`](../saddlellm/SaddleModeling.py#L1218)<br><sub>`from_pretrained_saddle(cls, path: str, map_location: Optional[str]=None) -> 'SaddleForCausalLM'`</sub> | method | 从检查点加载 `SaddleForCausalLM`，遵循预训练模型的目录契约。 | `os.path.join`, `os.path.isfile`, `FileNotFoundError`, `open`, `SaddleModelConfig`, `json.load`, `cls`, `check_torch_load_is_safe`, `Version`, `torch.__version__.split` |

## `saddlellm/SafeGenerate.py`

共 21 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`filter_toxic_texts`](../saddlellm/SafeGenerate.py#L75)<br><sub>`filter_toxic_texts(texts: List[str], lang: str='auto') -> List[str]`</sub> | function | 过滤有毒文本。 | `isinstance`, `text.strip`, `re.search`, `_has_repetition_problem`, `clean.append`, `logger.info`, `len` |
| [`filter_toxic_dataset`](../saddlellm/SafeGenerate.py#L109)<br><sub>`filter_toxic_dataset(dataset, text_column: str='text')`</sub> | function | 对 HuggingFace dataset 做有毒过滤。 | `dataset.filter` |
| [`filter_toxic_dataset._is_clean`](../saddlellm/SafeGenerate.py#L113)<br><sub>`_is_clean(example)`</sub> | nested function | 模块级清洗`is_clean`的局部回调/辅助逻辑。 | `example.get`, `re.search`, `_has_repetition_problem` |
| [`_has_repetition_problem`](../saddlellm/SafeGenerate.py#L126)<br><sub>`_has_repetition_problem(text: str) -> bool`</sub> | function | 检查是否有严重的重复问题。 | `len`, `re.search`, `re.split`, `s.strip`, `set` |
| [`SafeGenerate.__init__`](../saddlellm/SafeGenerate.py#L176)<br><sub>`__init__(self, model, tokenizer)`</sub> | method | 初始化 `SafeGenerate` 实例及其运行依赖。 | `next`, `model.parameters` |
| [`SafeGenerate.generate`](../saddlellm/SafeGenerate.py#L185)<br><sub>`generate(self, prompt: str, max_new_tokens: int=1024, temperature: float=0.7, top_p: float=0.9, enable_safety: bool=True, anti_repeat: bool=True, repetition_penalty: float=1.15, no_repeat_ngram_size: int=4, frequency_penalty: float=0.3, presence_penalty: float=0.3, use_contrastive_search: bool=False, contrastive_k: int=4, contrastive_alpha: float=0.6, max_retries: int=2) -> Dict`</sub> | method | 安全生成。组合多种策略避免骂人和车轱辘话。 | `range`, `to`, `self.tokenizer`, `gen_kwargs.pop`, `self.model.generate`, `self.tokenizer.decode`, `len`, `strip`, `self._check_repetition`, `self._check_toxicity` |
| [`SafeGenerate._check_repetition`](../saddlellm/SafeGenerate.py#L271)<br><sub>`_check_repetition(self, text: str) -> float`</sub> | method | 检测车轱辘话程度 (0-1, 越高越严重)。 | `len`, `text.split`, `tuple`, `range`, `set`, `re.search`, `re.split`, `s.strip`, `min` |
| [`SafeGenerate._check_toxicity`](../saddlellm/SafeGenerate.py#L320)<br><sub>`_check_toxicity(self, text: str) -> bool`</sub> | method | 快速检查是否有不当内容。 | `re.search` |
| [`SafeGenerate.generate_batch`](../saddlellm/SafeGenerate.py#L333)<br><sub>`generate_batch(self, prompts: List[str], **kwargs) -> List[Dict]`</sub> | method | 批量安全生成。 | `self.generate` |
| [`AntiRepeatGenerator.__init__`](../saddlellm/SafeGenerate.py#L377)<br><sub>`__init__(self, model, tokenizer)`</sub> | method | 初始化 `AntiRepeatGenerator` 实例及其运行依赖。 | `next`, `model.parameters` |
| [`AntiRepeatGenerator.generate`](../saddlellm/SafeGenerate.py#L383)<br><sub>`generate(self, prompt: str, method: str='auto', max_new_tokens: int=1024, temperature: float=0.7) -> str`</sub> | method | `AntiRepeatGenerator` 中生成`generate`的公开操作。 | `sum`, `p.numel`, `self.model.parameters`, `getattr`, `self.model.eval`, `to`, `self.tokenizer`, `torch.no_grad`, `gen_fn` |
| [`AntiRepeatGenerator._gen_repetition_penalty`](../saddlellm/SafeGenerate.py#L404)<br><sub>`_gen_repetition_penalty(self, inputs, max_tokens, temp)`</sub> | method | `AntiRepeatGenerator` 中实现`gen_repetition_penalty`的内部辅助逻辑。 | `self.model.generate`, `self._decode` |
| [`AntiRepeatGenerator._gen_frequency`](../saddlellm/SafeGenerate.py#L413)<br><sub>`_gen_frequency(self, inputs, max_tokens, temp)`</sub> | method | `AntiRepeatGenerator` 中实现`gen_frequency`的内部辅助逻辑。 | `self.model.generate`, `self._decode` |
| [`AntiRepeatGenerator._gen_no_repeat_ngram`](../saddlellm/SafeGenerate.py#L423)<br><sub>`_gen_no_repeat_ngram(self, inputs, max_tokens, temp)`</sub> | method | `AntiRepeatGenerator` 中实现`gen_no_repeat_ngram`的内部辅助逻辑。 | `self.model.generate`, `self._decode` |
| [`AntiRepeatGenerator._gen_contrastive`](../saddlellm/SafeGenerate.py#L433)<br><sub>`_gen_contrastive(self, inputs, max_tokens, temp)`</sub> | method | `AntiRepeatGenerator` 中实现`gen_contrastive`的内部辅助逻辑。 | `self.model.generate`, `self._decode` |
| [`AntiRepeatGenerator._gen_diverse_beam`](../saddlellm/SafeGenerate.py#L451)<br><sub>`_gen_diverse_beam(self, inputs, max_tokens, temp)`</sub> | method | `AntiRepeatGenerator` 中实现`gen_diverse_beam`的内部辅助逻辑。 | `self.model.generate`, `self._decode` |
| [`AntiRepeatGenerator._gen_default`](../saddlellm/SafeGenerate.py#L461)<br><sub>`_gen_default(self, inputs, max_tokens, temp)`</sub> | method | `AntiRepeatGenerator` 中实现`gen_default`的内部辅助逻辑。 | `self.model.generate`, `self._decode` |
| [`AntiRepeatGenerator._decode`](../saddlellm/SafeGenerate.py#L471)<br><sub>`_decode(self, outputs, inputs)`</sub> | method | `AntiRepeatGenerator` 中解码`decode`的内部辅助逻辑。 | `self.tokenizer.decode`, `len`, `strip` |
| [`OutputSafetyFilter.__init__`](../saddlellm/SafeGenerate.py#L488)<br><sub>`__init__(self, custom_blocklist: List[str]=None)`</sub> | method | 初始化 `OutputSafetyFilter` 实例及其运行依赖。 | — |
| [`OutputSafetyFilter.check`](../saddlellm/SafeGenerate.py#L492)<br><sub>`check(self, text: str) -> Dict`</sub> | method | 检查输出是否安全。 | `re.search`, `issues.append`, `item.lower`, `text.lower`, `_check_simple_repeat`, `len`, `match.group`, `match.start`, `match.end` |
| [`_check_simple_repeat`](../saddlellm/SafeGenerate.py#L533)<br><sub>`_check_simple_repeat(text: str) -> float`</sub> | function | 简单重复度检查。 | `len`, `range`, `set` |

## `saddlellm/ScalingLawAnalyzer.py`

共 14 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`PretrainRunPlan.to_dict`](../saddlellm/ScalingLawAnalyzer.py#L35)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `PretrainRunPlan` 转为可序列化字典。 | `asdict` |
| [`PretrainRunPlan.to_orchestrator_config`](../saddlellm/ScalingLawAnalyzer.py#L38)<br><sub>`to_orchestrator_config(self, corpus_sources: Sequence[Dict], output_dir: str, tokenizer_path: Optional[str]=None, learning_rate: float=0.0003, per_device_batch_size: int=1, stability_monitor: bool=True) -> Dict`</sub> | method | `PretrainRunPlan` 中实现配置的公开操作。 | `list`, `max` |
| [`ScalingLawAnalyzer.__init__`](../saddlellm/ScalingLawAnalyzer.py#L94)<br><sub>`__init__(self)`</sub> | method | 初始化 `ScalingLawAnalyzer` 实例及其运行依赖。 | — |
| [`ScalingLawAnalyzer.add_run`](../saddlellm/ScalingLawAnalyzer.py#L103)<br><sub>`add_run(self, model_name: str, params: int, tokens: int, loss: float, compute_flops: Optional[float]=None)`</sub> | method | 添加一次训练运行的数据点 | `self.runs.append`, `ScalingRun` |
| [`ScalingLawAnalyzer.add_runs_from_summaries`](../saddlellm/ScalingLawAnalyzer.py#L114)<br><sub>`add_runs_from_summaries(self, summary_paths: List[str])`</sub> | method | 从训练摘要 JSON 文件中批量导入 | `open`, `json.load`, `get`, `data.get`, `self.add_run`, `pretrain.get`, `eval_results.get`, `logger.warning` |
| [`ScalingLawAnalyzer.fit`](../saddlellm/ScalingLawAnalyzer.py#L132)<br><sub>`fit(self, alpha: float=0.34, beta: float=0.28) -> Dict`</sub> | method | 拟合缩放法则参数。 使用最小二乘法拟合 E, A, B。 | `len`, `logger.warning`, `sum`, `zip`, `abs`, `min`, `max`, `self._compute_r2` |
| [`ScalingLawAnalyzer.predict_loss`](../saddlellm/ScalingLawAnalyzer.py#L191)<br><sub>`predict_loss(self, params: int, tokens: int) -> float`</sub> | method | 预测给定模型大小和训练量的 loss | `self.fit` |
| [`ScalingLawAnalyzer.predict_chinchilla_optimal`](../saddlellm/ScalingLawAnalyzer.py#L197)<br><sub>`predict_chinchilla_optimal(self, compute_budget_flops: float) -> Tuple[int, int]`</sub> | method | 给定计算预算,返回 Chinchilla 最优的 (参数量, 训练 token 数)。 Chinchilla: N_opt ∝ C^0.5, D_opt ∝ C^0.5 | `int` |
| [`ScalingLawAnalyzer.recommend_next_run`](../saddlellm/ScalingLawAnalyzer.py#L210)<br><sub>`recommend_next_run(self, budget_flops: float) -> Dict`</sub> | method | 推荐下一个模型规模 | `self.predict_chinchilla_optimal`, `self.predict_loss` |
| [`ScalingLawAnalyzer.plan_pretrain_grid`](../saddlellm/ScalingLawAnalyzer.py#L222)<br><sub>`plan_pretrain_grid(self, model_names: Optional[Sequence[str]]=None, token_multipliers: Sequence[float]=(2.0, 5.0, 10.0, 20.0), global_batch_size: int=512, seq_length: int=2048, max_tokens: Optional[int]=None, max_params: Optional[int]=None, dtype: str='bf16') -> List[PretrainRunPlan]`</sub> | method | Generate a low-cost scratch-pretraining experiment grid. | `list`, `MODEL_SPECS.keys`, `max`, `int`, `min`, `math.ceil`, `get`, `ModelRegistry.estimate_memory`, `plans.append`, `PretrainRunPlan` |
| [`ScalingLawAnalyzer.export_pretrain_grid`](../saddlellm/ScalingLawAnalyzer.py#L273)<br><sub>`export_pretrain_grid(self, path: str, plans: List[PretrainRunPlan]) -> str`</sub> | method | Save a generated pretraining grid as JSON. | `os.makedirs`, `os.path.dirname`, `open`, `json.dump`, `p.to_dict` |
| [`ScalingLawAnalyzer.plot_scaling_curve`](../saddlellm/ScalingLawAnalyzer.py#L282)<br><sub>`plot_scaling_curve(self, save_path: Optional[str]=None)`</sub> | method | 绘制缩放曲线 (需要 matplotlib) | `logger.warning`, `self.fit`, `plt.subplots`, `ax.scatter`, `np.logspace`, `math.log10`, `min`, `max`, `int`, `np.median` |
| [`ScalingLawAnalyzer.generate_report`](../saddlellm/ScalingLawAnalyzer.py#L344)<br><sub>`generate_report(self) -> str`</sub> | method | 生成缩放分析报告 | `self.fit`, `len`, `sorted`, `self.predict_loss`, `lines.append`, `int`, `join` |
| [`ScalingLawAnalyzer._compute_r2`](../saddlellm/ScalingLawAnalyzer.py#L378)<br><sub>`_compute_r2(self) -> float`</sub> | method | 计算拟合的 R² | `sum`, `len`, `self.predict_loss` |

## `saddlellm/SimpleFlow.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SimpleFlowCompiler.is_simple_flow`](../saddlellm/SimpleFlow.py#L21)<br><sub>`is_simple_flow(cls, raw: Mapping[str, Any]) -> bool`</sub> | method | `SimpleFlowCompiler` 中实现流程的公开操作。 | — |
| [`SimpleFlowCompiler.compile`](../saddlellm/SimpleFlow.py#L25)<br><sub>`compile(cls, raw: Mapping[str, Any], base_dir: str \| None=None) -> Dict[str, Any]`</sub> | method | `SimpleFlowCompiler` 中编译`compile`的公开操作。 | `copy.deepcopy`, `dict`, `os.getcwd`, `cls._parse_flow`, `raw.get`, `cls._validate_flow`, `isinstance`, `model_raw.strip`, `strip`, `str` |
| [`SimpleFlowCompiler._parse_flow`](../saddlellm/SimpleFlow.py#L214)<br><sub>`_parse_flow(cls, flow: Any, raw: Mapping[str, Any]) -> List[str]`</sub> | method | `SimpleFlowCompiler` 中解析流程的内部辅助逻辑。 | `isinstance`, `flow.strip`, `ValueError`, `replace`, `flow.lower`, `lower`, `str`, `raw.get`, `normalized.split`, `list` |
| [`SimpleFlowCompiler._validate_flow`](../saddlellm/SimpleFlow.py#L227)<br><sub>`_validate_flow(cls, tokens: Iterable[str]) -> None`</sub> | method | `SimpleFlowCompiler` 中校验流程的内部辅助逻辑。 | `list`, `NotImplementedError`, `join`, `ValueError` |
| [`SimpleFlowCompiler._mapping`](../saddlellm/SimpleFlow.py#L242)<br><sub>`_mapping(value: Any, name: str) -> Dict[str, Any]`</sub> | method | `SimpleFlowCompiler` 中实现`mapping`的内部辅助逻辑。 | `isinstance`, `ValueError`, `dict` |
| [`SimpleFlowCompiler._required_data_path`](../saddlellm/SimpleFlow.py#L250)<br><sub>`_required_data_path(data: Mapping[str, Any], key: str) -> str`</sub> | method | `SimpleFlowCompiler` 中实现数据、路径的内部辅助逻辑。 | `data.get`, `isinstance`, `value.strip`, `ValueError` |
| [`SimpleFlowCompiler._compile_evaluation`](../saddlellm/SimpleFlow.py#L257)<br><sub>`_compile_evaluation(evaluation: Mapping[str, Any], enabled: bool) -> Dict[str, Any]`</sub> | method | `SimpleFlowCompiler` 中编译`compile_evaluation`的内部辅助逻辑。 | `evaluation.get`, `isinstance`, `str`, `ValueError`, `SimpleFlowCompiler._mapping`, `gate.get`, `int`, `bool`, `copy.deepcopy`, `dict` |

## `saddlellm/SmokeTestRunner.py`

共 20 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SmokeCommandResult.to_dict`](../saddlellm/SmokeTestRunner.py#L27)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `SmokeCommandResult` 转为可序列化字典。 | `asdict` |
| [`SmokeTestResult.to_dict`](../saddlellm/SmokeTestRunner.py#L41)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `SmokeTestResult` 转为可序列化字典。 | `asdict`, `item.to_dict` |
| [`SmokeTestRunner.__init__`](../saddlellm/SmokeTestRunner.py#L50)<br><sub>`__init__(self, work_dir: str='build/smoke_e2e_auto', run_training: bool=True, clean: bool=True, run_doctor: bool=True, timeout_seconds: int=180, use_subprocess: bool=True)`</sub> | method | 初始化 `SmokeTestRunner` 实例及其运行依赖。 | `Path` |
| [`SmokeTestRunner.run`](../saddlellm/SmokeTestRunner.py#L66)<br><sub>`run(self) -> SmokeTestResult`</sub> | method | `SmokeTestRunner` 中执行`run`的公开操作。 | `time.time`, `self.work_dir.exists`, `shutil.rmtree`, `self._prepare_fixtures`, `commands.append`, `self._run_command`, `commands.extend`, `str`, `self._data_path`, `self._image_dir` |
| [`SmokeTestRunner._run_command`](../saddlellm/SmokeTestRunner.py#L152)<br><sub>`_run_command(self, name: str, args: List[str], expected_returncode: int=0, module: bool=True) -> SmokeCommandResult`</sub> | method | `SmokeTestRunner` 中执行`run_command`的内部辅助逻辑。 | `command.extend`, `time.time`, `self._run_in_process`, `subprocess.run`, `os.getcwd`, `completed.stdout.splitlines`, `line.strip`, `SmokeCommandResult`, `round` |
| [`SmokeTestRunner._run_in_process`](../saddlellm/SmokeTestRunner.py#L179)<br><sub>`_run_in_process(self, name: str, command: List[str], args: List[str], expected_returncode: int, module: bool, start: float) -> SmokeCommandResult`</sub> | method | `SmokeTestRunner` 中执行`run_in_process`的内部辅助逻辑。 | `io.StringIO`, `contextlib.redirect_stdout`, `contextlib.redirect_stderr`, `int`, `main`, `compileall.compile_dir`, `isinstance`, `print`, `type`, `gc.collect` |
| [`SmokeTestRunner._prepare_fixtures`](../saddlellm/SmokeTestRunner.py#L225)<br><sub>`_prepare_fixtures(self) -> None`</sub> | method | `SmokeTestRunner` 中准备`prepare_fixtures`的内部辅助逻辑。 | `mkdir`, `self._data_dir`, `self._image_dir`, `self._write_tiny_model`, `self._write_data`, `self._write_configs` |
| [`SmokeTestRunner._write_tiny_model`](../saddlellm/SmokeTestRunner.py#L232)<br><sub>`_write_tiny_model(self) -> None`</sub> | method | `SmokeTestRunner` 中写入模型的内部辅助逻辑。 | `Tokenizer`, `WordLevel`, `Whitespace`, `PreTrainedTokenizerFast`, `GPT2Config`, `len`, `GPT2LMHeadModel`, `mkdir`, `self._model_dir`, `model.save_pretrained` |
| [`SmokeTestRunner._write_data`](../saddlellm/SmokeTestRunner.py#L287)<br><sub>`_write_data(self) -> None`</sub> | method | `SmokeTestRunner` 中写入数据的内部辅助逻辑。 | `self._write_jsonl`, `self._data_path`, `base64.b64decode`, `open`, `self._image_dir`, `f.write` |
| [`SmokeTestRunner._write_configs`](../saddlellm/SmokeTestRunner.py#L334)<br><sub>`_write_configs(self) -> None`</sub> | method | `SmokeTestRunner` 中写入`write_configs`的内部辅助逻辑。 | `self._base_config`, `self._preference_config`, `str`, `self._data_path`, `self._image_dir`, `configs.items`, `open`, `self._config_path`, `yaml.safe_dump` |
| [`SmokeTestRunner._base_config`](../saddlellm/SmokeTestRunner.py#L392)<br><sub>`_base_config(self, experiment: str, output_name: str, stages: List[str]) -> Dict`</sub> | method | `SmokeTestRunner` 中实现配置的内部辅助逻辑。 | `str`, `self._model_dir` |
| [`SmokeTestRunner._preference_config`](../saddlellm/SmokeTestRunner.py#L407)<br><sub>`_preference_config(self, experiment: str, output_name: str, method: str, data_name: str, batch_size: int) -> Dict`</sub> | method | `SmokeTestRunner` 中实现配置的内部辅助逻辑。 | `self._base_config`, `str`, `self._data_path` |
| [`SmokeTestRunner._write_jsonl`](../saddlellm/SmokeTestRunner.py#L436)<br><sub>`_write_jsonl(path: Path, rows: List[Dict]) -> None`</sub> | method | `SmokeTestRunner` 中写入`write_jsonl`的内部辅助逻辑。 | `open`, `f.write`, `json.dumps` |
| [`SmokeTestRunner._check_training_artifacts`](../saddlellm/SmokeTestRunner.py#L442)<br><sub>`_check_training_artifacts(self) -> SmokeCommandResult`</sub> | method | `SmokeTestRunner` 中检查训练的内部辅助逻辑。 | `time.time`, `expected.items`, `path.exists`, `open`, `json.load`, `summary.get`, `details.append`, `missing.append`, `SmokeCommandResult`, `round` |
| [`SmokeTestRunner._model_dir`](../saddlellm/SmokeTestRunner.py#L473)<br><sub>`_model_dir(self) -> Path`</sub> | method | `SmokeTestRunner` 中实现模型的内部辅助逻辑。 | — |
| [`SmokeTestRunner._data_dir`](../saddlellm/SmokeTestRunner.py#L476)<br><sub>`_data_dir(self) -> Path`</sub> | method | `SmokeTestRunner` 中实现数据的内部辅助逻辑。 | — |
| [`SmokeTestRunner._image_dir`](../saddlellm/SmokeTestRunner.py#L479)<br><sub>`_image_dir(self) -> Path`</sub> | method | `SmokeTestRunner` 中实现图像的内部辅助逻辑。 | `self._data_dir` |
| [`SmokeTestRunner._data_path`](../saddlellm/SmokeTestRunner.py#L482)<br><sub>`_data_path(self, name: str) -> Path`</sub> | method | `SmokeTestRunner` 中实现数据、路径的内部辅助逻辑。 | `self._data_dir` |
| [`SmokeTestRunner._config_path`](../saddlellm/SmokeTestRunner.py#L485)<br><sub>`_config_path(self, name: str) -> Path`</sub> | method | `SmokeTestRunner` 中实现配置、路径的内部辅助逻辑。 | — |
| [`run_smoke_tests`](../saddlellm/SmokeTestRunner.py#L489)<br><sub>`run_smoke_tests(work_dir: str='build/smoke_e2e_auto', run_training: bool=True, clean: bool=True, run_doctor: bool=True, timeout_seconds: int=180, use_subprocess: bool=True) -> Dict`</sub> | function | 模块级执行`run_smoke_tests`的公开操作。 | `to_dict`, `run`, `SmokeTestRunner` |

## `saddlellm/SpatialAPI.py`

共 27 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SpatialPlanRequest.finite_coordinate`](../saddlellm/SpatialAPI.py#L77)<br><sub>`finite_coordinate(cls, value: Coordinate) -> Coordinate`</sub> | method | `SpatialPlanRequest` 中实现`finite_coordinate`的公开操作。 | `all`, `float`, `abs`, `ValueError` |
| [`SpatialStudioSettings.from_env`](../saddlellm/SpatialAPI.py#L95)<br><sub>`from_env(cls) -> 'SpatialStudioSettings'`</sub> | method | `SpatialStudioSettings` 中实现`from_env`的公开操作。 | `resolve`, `Path`, `cls`, `os.environ.get`, `str`, `int` |
| [`SpatialStudioSettings.__post_init__`](../saddlellm/SpatialAPI.py#L114)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `SpatialStudioSettings` 创建后校验并规范化字段。 | `str`, `resolve`, `Path`, `ValueError` |
| [`SpatialStudioError.__init__`](../saddlellm/SpatialAPI.py#L126)<br><sub>`__init__(self, message: str, status_code: int=400, code: str='invalid_request')`</sub> | method | 初始化 `SpatialStudioError` 实例及其运行依赖。 | `__init__`, `super`, `int`, `str` |
| [`SpatialStudioService.__init__`](../saddlellm/SpatialAPI.py#L135)<br><sub>`__init__(self, settings: Optional[SpatialStudioSettings]=None, analyzer_factory: Optional[Callable[[str], Any]]=None, runtime_factory: Optional[Callable[[str], Any]]=None) -> None`</sub> | method | 初始化 `SpatialStudioService` 实例及其运行依赖。 | `SpatialStudioSettings.from_env`, `resolve`, `Path`, `self.workspace.mkdir`, `threading.Lock`, `SpatialObservationEncoder`, `SpatialTensorObservationEncoder` |
| [`SpatialStudioService.capabilities`](../saddlellm/SpatialAPI.py#L155)<br><sub>`capabilities(self) -> Dict[str, Any]`</sub> | method | `SpatialStudioService` 中实现`capabilities`的公开操作。 | `bool`, `_display_model_name`, `self.observation_encoder.describe`, `self.tensor_observation_encoder.describe` |
| [`SpatialStudioService.plan_bytes`](../saddlellm/SpatialAPI.py#L187)<br><sub>`plan_bytes(self, payload: bytes, request: SpatialPlanRequest, original_filename: str='upload') -> Dict[str, Any]`</sub> | method | `SpatialStudioService` 中规划计划的公开操作。 | `len`, `SpatialStudioError`, `uuid.uuid4`, `final_directory.mkdir`, `self._decode_image`, `self._validate_coordinates`, `self._get_analyzer`, `TopDownMapExtractor`, `MapExtractionConfig`, `GridPathPlanner` |
| [`SpatialStudioService.demo`](../saddlellm/SpatialAPI.py#L354)<br><sub>`demo(self) -> Dict[str, Any]`</sub> | method | `SpatialStudioService` 中实现`demo`的公开操作。 | `Path`, `demo_path.is_absolute`, `Path.cwd`, `demo_path.is_file`, `SpatialStudioError`, `SpatialPlanRequest`, `self.plan_bytes`, `demo_path.read_bytes` |
| [`SpatialStudioService.job_response`](../saddlellm/SpatialAPI.py#L382)<br><sub>`job_response(self, job_id: str) -> Dict[str, Any]`</sub> | method | `SpatialStudioService` 中实现响应的公开操作。 | `self._job_directory`, `path.is_file`, `SpatialStudioError`, `json.loads`, `path.read_text` |
| [`SpatialStudioService.artifact_path`](../saddlellm/SpatialAPI.py#L388)<br><sub>`artifact_path(self, job_id: str, artifact: str) -> Tuple[Path, str]`</sub> | method | `SpatialStudioService` 中实现产物、路径的公开操作。 | `SpatialStudioError`, `self._job_directory`, `path.is_file` |
| [`SpatialStudioService.delete_job`](../saddlellm/SpatialAPI.py#L397)<br><sub>`delete_job(self, job_id: str) -> None`</sub> | method | `SpatialStudioService` 中删除`delete_job`的公开操作。 | `self._job_directory`, `directory.is_dir`, `SpatialStudioError`, `shutil.rmtree`, `self._demo_response.get` |
| [`SpatialStudioService._decode_image`](../saddlellm/SpatialAPI.py#L405)<br><sub>`_decode_image(self, payload: bytes, destination: Path) -> Tuple[int, int]`</sub> | method | `SpatialStudioService` 中解码图像的内部辅助逻辑。 | `SpatialStudioError`, `Image.open`, `io.BytesIO`, `source.seek`, `save`, `source.convert`, `int` |
| [`SpatialStudioService._validate_coordinates`](../saddlellm/SpatialAPI.py#L428)<br><sub>`_validate_coordinates(request: SpatialPlanRequest, image_size: Tuple[int, int]) -> None`</sub> | method | `SpatialStudioService` 中校验`validate_coordinates`的内部辅助逻辑。 | `SpatialStudioError` |
| [`SpatialStudioService._get_analyzer`](../saddlellm/SpatialAPI.py#L438)<br><sub>`_get_analyzer(self, backend: str) -> Optional[Any]`</sub> | method | `SpatialStudioService` 中读取`get_analyzer`的内部辅助逻辑。 | `SpatialStudioError`, `QwenVLSpatialAnalyzer.from_pretrained`, `factory` |
| [`SpatialStudioService._get_runtime`](../saddlellm/SpatialAPI.py#L455)<br><sub>`_get_runtime(self, requested: bool) -> Optional[Any]`</sub> | method | `SpatialStudioService` 中读取运行时的内部辅助逻辑。 | `SpatialStudioError`, `self._runtime_factory`, `WorldModelRuntime.from_pretrained` |
| [`SpatialStudioService._job_directory`](../saddlellm/SpatialAPI.py#L479)<br><sub>`_job_directory(self, job_id: str) -> Path`</sub> | method | `SpatialStudioService` 中实现`job_directory`的内部辅助逻辑。 | `_JOB_PATTERN.fullmatch`, `str`, `SpatialStudioError`, `resolve` |
| [`create_spatial_studio_app`](../saddlellm/SpatialAPI.py#L488)<br><sub>`create_spatial_studio_app(settings: Optional[SpatialStudioSettings]=None, service: Optional[SpatialStudioService]=None) -> FastAPI`</sub> | function | 模块级创建`create_spatial_studio_app`的公开操作。 | `SpatialStudioSettings.from_env`, `SpatialStudioService`, `FastAPI`, `app.add_middleware`, `WorldAgentSettings`, `str`, `WorldAgentRuntime`, `service._get_analyzer`, `service._get_runtime`, `install_world_agent_routes` |
| [`create_spatial_studio_app.handle_spatial_error`](../saddlellm/SpatialAPI.py#L539)<br><sub>`async handle_spatial_error(_request: Any, error: SpatialStudioError) -> JSONResponse`</sub> | nested function | 模块级处理`handle_spatial_error`的局部回调/辅助逻辑。 | `JSONResponse`, `str` |
| [`create_spatial_studio_app.health`](../saddlellm/SpatialAPI.py#L546)<br><sub>`async health() -> Dict[str, Any]`</sub> | nested function | 模块级实现`health`的局部回调/辅助逻辑。 | — |
| [`create_spatial_studio_app.capabilities`](../saddlellm/SpatialAPI.py#L550)<br><sub>`async capabilities() -> Dict[str, Any]`</sub> | nested function | 模块级实现`capabilities`的局部回调/辅助逻辑。 | `service.capabilities` |
| [`create_spatial_studio_app.demo`](../saddlellm/SpatialAPI.py#L554)<br><sub>`async demo() -> Dict[str, Any]`</sub> | nested function | 模块级实现`demo`的局部回调/辅助逻辑。 | `run_in_threadpool` |
| [`create_spatial_studio_app.plan`](../saddlellm/SpatialAPI.py#L558)<br><sub>`async plan(image: UploadFile=File(...), request: str=Form(...)) -> Dict[str, Any]`</sub> | nested function | 模块级规划计划的局部回调/辅助逻辑。 | `SpatialPlanRequest.model_validate_json`, `HTTPException`, `image.read`, `run_in_threadpool` |
| [`create_spatial_studio_app.job`](../saddlellm/SpatialAPI.py#L576)<br><sub>`async job(job_id: str) -> Dict[str, Any]`</sub> | nested function | 模块级实现`job`的局部回调/辅助逻辑。 | `service.job_response` |
| [`create_spatial_studio_app.artifact`](../saddlellm/SpatialAPI.py#L580)<br><sub>`async artifact(job_id: str, artifact: str) -> FileResponse`</sub> | nested function | 模块级实现产物的局部回调/辅助逻辑。 | `service.artifact_path`, `FileResponse` |
| [`create_spatial_studio_app.delete_job`](../saddlellm/SpatialAPI.py#L589)<br><sub>`async delete_job(job_id: str) -> Dict[str, str]`</sub> | nested function | 模块级删除`delete_job`的局部回调/辅助逻辑。 | `service.delete_job` |
| [`create_spatial_studio_app.root`](../saddlellm/SpatialAPI.py#L598)<br><sub>`async root() -> Dict[str, Any]`</sub> | nested function | 模块级实现`root`的局部回调/辅助逻辑。 | — |
| [`_display_model_name`](../saddlellm/SpatialAPI.py#L608)<br><sub>`_display_model_name(path: Optional[str]) -> Optional[str]`</sub> | function | 模块级实现模型的内部辅助逻辑。 | `rstrip`, `str`, `os.path.basename` |

## `saddlellm/SpatialPerception.py`

共 16 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SpatialEntity.__post_init__`](../saddlellm/SpatialPerception.py#L24)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `SpatialEntity` 创建后校验并规范化字段。 | `tuple`, `float`, `len`, `ValueError` |
| [`SpatialEntity.center`](../saddlellm/SpatialPerception.py#L32)<br><sub>`center(self) -> Tuple[float, float]`</sub> | method | `SpatialEntity` 中实现`center`的公开操作。 | — |
| [`SpatialEntity.from_dict`](../saddlellm/SpatialPerception.py#L37)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'SpatialEntity'`</sub> | method | 从字典解析并创建 `SpatialEntity`。 | `dict`, `values.setdefault`, `values.get`, `fields`, `cls`, `values.items` |
| [`SpatialAnalysis.__post_init__`](../saddlellm/SpatialPerception.py#L59)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `SpatialAnalysis` 创建后校验并规范化字段。 | `isinstance`, `SpatialEntity.from_dict`, `float`, `ValueError` |
| [`SpatialAnalysis.from_dict`](../saddlellm/SpatialPerception.py#L69)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'SpatialAnalysis'`</sub> | method | 从字典解析并创建 `SpatialAnalysis`。 | `dict`, `fields`, `cls`, `values.items` |
| [`SpatialAnalysis.to_dict`](../saddlellm/SpatialPerception.py#L74)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `SpatialAnalysis` 转为可序列化字典。 | `asdict` |
| [`MapExtractionConfig.__post_init__`](../saddlellm/SpatialPerception.py#L90)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `MapExtractionConfig` 创建后校验并规范化字段。 | `ValueError` |
| [`TopDownMapExtractor.__init__`](../saddlellm/SpatialPerception.py#L106)<br><sub>`__init__(self, config: Optional[MapExtractionConfig]=None) -> None`</sub> | method | 初始化 `TopDownMapExtractor` 实例及其运行依赖。 | `MapExtractionConfig` |
| [`TopDownMapExtractor.extract`](../saddlellm/SpatialPerception.py#L109)<br><sub>`extract(self, image_path: str) -> OccupancyGrid`</sub> | method | `TopDownMapExtractor` 中提取`extract`的公开操作。 | `ImportError`, `Image.open`, `source.convert`, `max`, `int`, `round`, `getattr`, `image.resize`, `np.asarray`, `astype` |
| [`CallableSpatialAnalyzer.__init__`](../saddlellm/SpatialPerception.py#L166)<br><sub>`__init__(self, analyze_fn: Callable[[str, str], Union[SpatialAnalysis, Dict[str, Any]]])`</sub> | method | 初始化 `CallableSpatialAnalyzer` 实例及其运行依赖。 | — |
| [`CallableSpatialAnalyzer.analyze`](../saddlellm/SpatialPerception.py#L169)<br><sub>`analyze(self, image_path: str, instruction: str='') -> SpatialAnalysis`</sub> | method | `CallableSpatialAnalyzer` 中分析`analyze`的公开操作。 | `self.analyze_fn`, `isinstance`, `SpatialAnalysis.from_dict` |
| [`QwenVLSpatialAnalyzer.__init__`](../saddlellm/SpatialPerception.py#L204)<br><sub>`__init__(self, model: Any, processor: Any, max_new_tokens: int=1200) -> None`</sub> | method | 初始化 `QwenVLSpatialAnalyzer` 实例及其运行依赖。 | `int` |
| [`QwenVLSpatialAnalyzer.from_pretrained`](../saddlellm/SpatialPerception.py#L215)<br><sub>`from_pretrained(cls, model_name_or_path: str, device_map: str='auto', torch_dtype: str='auto', trust_remote_code: bool=True, max_new_tokens: int=1200) -> 'QwenVLSpatialAnalyzer'`</sub> | method | 从检查点加载 `QwenVLSpatialAnalyzer`，遵循预训练模型的目录契约。 | `ImportError`, `eval`, `ModelClass.from_pretrained`, `AutoProcessor.from_pretrained`, `cls` |
| [`QwenVLSpatialAnalyzer.analyze`](../saddlellm/SpatialPerception.py#L245)<br><sub>`analyze(self, image_path: str, instruction: str='') -> SpatialAnalysis`</sub> | method | `QwenVLSpatialAnalyzer` 中分析`analyze`的公开操作。 | `ImportError`, `str`, `convert`, `Image.open`, `self.processor.apply_chat_template`, `self.processor`, `next`, `self.model.parameters`, `hasattr`, `value.to` |
| [`_extract_json_object`](../saddlellm/SpatialPerception.py#L289)<br><sub>`_extract_json_object(text: str) -> Dict[str, Any]`</sub> | function | 模块级提取`extract_json_object`的内部辅助逻辑。 | `strip`, `str`, `re.search`, `fenced.group`, `json.loads`, `candidate.find`, `candidate.rfind`, `ValueError`, `isinstance` |
| [`_dilate`](../saddlellm/SpatialPerception.py#L306)<br><sub>`_dilate(mask: np.ndarray, radius: int) -> np.ndarray`</sub> | function | 模块级实现`dilate`的内部辅助逻辑。 | `mask.copy`, `range`, `max`, `min` |

## `saddlellm/SpatialPlanner.py`

共 26 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`OccupancyGrid.__post_init__`](../saddlellm/SpatialPlanner.py#L31)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `OccupancyGrid` 创建后校验并规范化字段。 | `np.asarray`, `ValueError`, `all`, `np.isin`, `tuple`, `int` |
| [`OccupancyGrid.width`](../saddlellm/SpatialPlanner.py#L43)<br><sub>`width(self) -> int`</sub> | method | `OccupancyGrid` 中实现`width`的公开操作。 | `int` |
| [`OccupancyGrid.height`](../saddlellm/SpatialPlanner.py#L47)<br><sub>`height(self) -> int`</sub> | method | `OccupancyGrid` 中实现`height`的公开操作。 | `int` |
| [`OccupancyGrid.in_bounds`](../saddlellm/SpatialPlanner.py#L50)<br><sub>`in_bounds(self, point: GridPoint) -> bool`</sub> | method | `OccupancyGrid` 中实现`in_bounds`的公开操作。 | — |
| [`OccupancyGrid.cell`](../saddlellm/SpatialPlanner.py#L54)<br><sub>`cell(self, point: GridPoint) -> int`</sub> | method | `OccupancyGrid` 中实现`cell`的公开操作。 | `self.in_bounds`, `IndexError`, `int` |
| [`OccupancyGrid.traversable`](../saddlellm/SpatialPlanner.py#L59)<br><sub>`traversable(self, point: GridPoint, allow_unknown: bool=False) -> bool`</sub> | method | `OccupancyGrid` 中实现`traversable`的公开操作。 | `self.in_bounds`, `self.cell` |
| [`OccupancyGrid.scale_from_source`](../saddlellm/SpatialPlanner.py#L65)<br><sub>`scale_from_source(self, point: Sequence[float]) -> GridPoint`</sub> | method | `OccupancyGrid` 中实现数据源的公开操作。 | `len`, `ValueError`, `float`, `max`, `min`, `int`, `round` |
| [`OccupancyGrid.scale_to_source`](../saddlellm/SpatialPlanner.py#L78)<br><sub>`scale_to_source(self, point: GridPoint) -> Tuple[float, float]`</sub> | method | `OccupancyGrid` 中实现数据源的公开操作。 | `float`, `max` |
| [`OccupancyGrid.to_dict`](../saddlellm/SpatialPlanner.py#L88)<br><sub>`to_dict(self, include_cells: bool=False) -> Dict[str, Any]`</sub> | method | 把 `OccupancyGrid` 转为可序列化字典。 | `np.unique`, `str`, `int`, `zip`, `list`, `cell_counts.get`, `self.cells.tolist` |
| [`SpatialPlannerConfig.__post_init__`](../saddlellm/SpatialPlanner.py#L118)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `SpatialPlannerConfig` 创建后校验并规范化字段。 | `ValueError` |
| [`RouteCandidate.to_dict`](../saddlellm/SpatialPlanner.py#L146)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `RouteCandidate` 转为可序列化字典。 | `asdict`, `int` |
| [`GridPathPlanner.__init__`](../saddlellm/SpatialPlanner.py#L155)<br><sub>`__init__(self, config: Optional[SpatialPlannerConfig]=None) -> None`</sub> | method | 初始化 `GridPathPlanner` 实例及其运行依赖。 | `SpatialPlannerConfig` |
| [`GridPathPlanner.plan`](../saddlellm/SpatialPlanner.py#L158)<br><sub>`plan(self, grid: OccupancyGrid, start: GridPoint, goal: GridPoint, route_count: Optional[int]=None) -> List[RouteCandidate]`</sub> | method | `GridPathPlanner` 中规划计划的公开操作。 | `_point`, `grid.traversable`, `ValueError`, `int`, `self.clearance_map`, `self._diverse_paths`, `self._yen_paths`, `self._route_metrics`, `len`, `enumerate` |
| [`GridPathPlanner._diverse_paths`](../saddlellm/SpatialPlanner.py#L192)<br><sub>`_diverse_paths(self, grid: OccupancyGrid, start: GridPoint, goal: GridPoint, route_count: int, clearance: np.ndarray) -> List[List[GridPoint]]`</sub> | method | Find practical alternatives by discouraging reused route corridors. | `self._astar`, `ValueError`, `tuple`, `max`, `self._path_cost`, `np.zeros`, `self._add_route_influence`, `len`, `self._path_overlap`, `proposals.append` |
| [`GridPathPlanner._add_route_influence`](../saddlellm/SpatialPlanner.py#L255)<br><sub>`_add_route_influence(self, influence: np.ndarray, path: Sequence[GridPoint]) -> None`</sub> | method | `GridPathPlanner` 中添加路线的内部辅助逻辑。 | `range`, `abs` |
| [`GridPathPlanner._path_overlap`](../saddlellm/SpatialPlanner.py#L272)<br><sub>`_path_overlap(first: Sequence[GridPoint], second: Sequence[GridPoint]) -> float`</sub> | method | `GridPathPlanner` 中实现路径的内部辅助逻辑。 | `set`, `max`, `min`, `len` |
| [`GridPathPlanner.shortest_path`](../saddlellm/SpatialPlanner.py#L278)<br><sub>`shortest_path(self, grid: OccupancyGrid, start: GridPoint, goal: GridPoint) -> List[GridPoint]`</sub> | method | `GridPathPlanner` 中实现路径的公开操作。 | `self._astar`, `_point`, `self.clearance_map` |
| [`GridPathPlanner.clearance_map`](../saddlellm/SpatialPlanner.py#L286)<br><sub>`clearance_map(self, grid: OccupancyGrid) -> np.ndarray`</sub> | method | `GridPathPlanner` 中实现`clearance_map`的公开操作。 | `np.full`, `deque`, `np.argwhere`, `queue.append`, `int`, `distances.fill`, `float`, `max`, `queue.popleft` |
| [`GridPathPlanner._yen_paths`](../saddlellm/SpatialPlanner.py#L309)<br><sub>`_yen_paths(self, grid: OccupancyGrid, start: GridPoint, goal: GridPoint, route_count: int, clearance: np.ndarray) -> List[List[GridPoint]]`</sub> | method | `GridPathPlanner` 中实现`yen_paths`的内部辅助逻辑。 | `self._astar`, `ValueError`, `tuple`, `set`, `count`, `len`, `range`, `banned_edges.add`, `candidate_keys.add`, `heapq.heappush` |
| [`GridPathPlanner._astar`](../saddlellm/SpatialPlanner.py#L363)<br><sub>`_astar(self, grid: OccupancyGrid, start: GridPoint, goal: GridPoint, clearance: np.ndarray, banned_nodes: Optional[Set[GridPoint]]=None, banned_edges: Optional[Set[GridEdge]]=None, node_penalties: Optional[np.ndarray]=None) -> List[GridPoint]`</sub> | method | `GridPathPlanner` 中实现`astar`的内部辅助逻辑。 | `set`, `count`, `heapq.heappush`, `self._heuristic`, `next`, `heapq.heappop`, `self._reconstruct`, `self._neighbors`, `grid.cell`, `float` |
| [`GridPathPlanner._neighbors`](../saddlellm/SpatialPlanner.py#L411)<br><sub>`_neighbors(self, grid: OccupancyGrid, point: GridPoint) -> Iterable[Tuple[GridPoint, float]]`</sub> | method | `GridPathPlanner` 中实现`neighbors`的内部辅助逻辑。 | `math.sqrt`, `movements.extend`, `grid.traversable` |
| [`GridPathPlanner._heuristic`](../saddlellm/SpatialPlanner.py#L444)<br><sub>`_heuristic(self, point: GridPoint, goal: GridPoint) -> float`</sub> | method | `GridPathPlanner` 中实现`heuristic`的内部辅助逻辑。 | `abs`, `float`, `max`, `math.sqrt`, `min` |
| [`GridPathPlanner._path_cost`](../saddlellm/SpatialPlanner.py#L451)<br><sub>`_path_cost(self, grid: OccupancyGrid, path: Sequence[GridPoint], clearance: np.ndarray) -> float`</sub> | method | `GridPathPlanner` 中实现路径的内部辅助逻辑。 | `zip`, `math.sqrt`, `grid.cell`, `float` |
| [`GridPathPlanner._route_metrics`](../saddlellm/SpatialPlanner.py#L472)<br><sub>`_route_metrics(self, grid: OccupancyGrid, path: List[GridPoint], clearance: np.ndarray, label: str) -> RouteCandidate`</sub> | method | `GridPathPlanner` 中实现路线、指标的内部辅助逻辑。 | `np.asarray`, `int`, `math.copysign`, `zip`, `sum`, `float`, `np.mean`, `math.sqrt`, `RouteCandidate`, `self._path_cost` |
| [`GridPathPlanner._reconstruct`](../saddlellm/SpatialPlanner.py#L507)<br><sub>`_reconstruct(came_from: Dict[GridPoint, GridPoint], current: GridPoint) -> List[GridPoint]`</sub> | method | `GridPathPlanner` 中实现`reconstruct`的内部辅助逻辑。 | `path.append`, `path.reverse` |
| [`_point`](../saddlellm/SpatialPlanner.py#L518)<br><sub>`_point(value: Sequence[int]) -> GridPoint`</sub> | function | 模块级实现`point`的内部辅助逻辑。 | `len`, `ValueError`, `int` |

## `saddlellm/SpatialVisualization.py`

共 13 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`save_spatial_plan_json`](../saddlellm/SpatialVisualization.py#L27)<br><sub>`save_spatial_plan_json(result: SpatialPlanResult, output_path: str, include_grid: bool=False) -> str`</sub> | function | 模块级保存计划的公开操作。 | `os.makedirs`, `os.path.dirname`, `os.path.abspath`, `open`, `json.dump`, `result.to_dict`, `file.write` |
| [`render_spatial_plan_html`](../saddlellm/SpatialVisualization.py#L44)<br><sub>`render_spatial_plan_html(result: SpatialPlanResult, output_path: str) -> str`</sub> | function | Write an offline, responsive route overlay with an accessible table. | `_file_data_uri`, `_occupancy_mask_data_uri`, `join`, `_route_svg`, `enumerate`, `_route_control`, `_route_table_row`, `_entity_svg`, `html.escape`, `str` |
| [`render_spatial_plan_png`](../saddlellm/SpatialVisualization.py#L246)<br><sub>`render_spatial_plan_png(result: SpatialPlanResult, output_path: str, width: int=1280) -> str`</sub> | function | Render a shareable static preview of the spatial plan. | `ImportError`, `ValueError`, `max`, `int`, `round`, `Image.new`, `ImageDraw.Draw`, `min`, `_load_font`, `draw.text` |
| [`render_spatial_plan_png.screen_point`](../saddlellm/SpatialVisualization.py#L316)<br><sub>`screen_point(point: Any) -> tuple[float, float]`</sub> | nested function | 模块级实现`screen_point`的局部回调/辅助逻辑。 | `float` |
| [`save_occupancy_mask_png`](../saddlellm/SpatialVisualization.py#L406)<br><sub>`save_occupancy_mask_png(grid: OccupancyGrid, output_path: str) -> str`</sub> | function | Save a transparent blocked/unknown overlay for browser map rendering. | `ImportError`, `np.zeros`, `os.makedirs`, `os.path.dirname`, `os.path.abspath`, `save`, `Image.fromarray` |
| [`_load_font`](../saddlellm/SpatialVisualization.py#L421)<br><sub>`_load_font(image_font: Any, size: int, bold: bool=False) -> Any`</sub> | function | 模块级加载`load_font`的内部辅助逻辑。 | `image_font.truetype`, `image_font.load_default` |
| [`_clip_text`](../saddlellm/SpatialVisualization.py#L431)<br><sub>`_clip_text(draw: Any, value: str, font: Any, maximum_width: int) -> str`</sub> | function | 模块级实现`clip_text`的内部辅助逻辑。 | `str`, `draw.textlength` |
| [`_route_svg`](../saddlellm/SpatialVisualization.py#L441)<br><sub>`_route_svg(index: int, route: RouteCandidate) -> str`</sub> | function | 模块级实现路线的内部辅助逻辑。 | `join`, `len` |
| [`_route_control`](../saddlellm/SpatialVisualization.py#L454)<br><sub>`_route_control(index: int, route: RouteCandidate) -> str`</sub> | function | 模块级实现路线的内部辅助逻辑。 | `html.escape` |
| [`_route_table_row`](../saddlellm/SpatialVisualization.py#L471)<br><sub>`_route_table_row(index: int, route: RouteCandidate) -> str`</sub> | function | 模块级实现路线的内部辅助逻辑。 | `html.escape` |
| [`_entity_svg`](../saddlellm/SpatialVisualization.py#L481)<br><sub>`_entity_svg(entity: Any, width: int, height: int) -> str`</sub> | function | 模块级实现`entity_svg`的内部辅助逻辑。 | `max`, `html.escape` |
| [`_file_data_uri`](../saddlellm/SpatialVisualization.py#L494)<br><sub>`_file_data_uri(path: str) -> str`</sub> | function | 模块级实现数据的内部辅助逻辑。 | `ImportError`, `Image.open`, `BytesIO`, `save`, `image.convert`, `decode`, `base64.b64encode`, `buffer.getvalue` |
| [`_occupancy_mask_data_uri`](../saddlellm/SpatialVisualization.py#L508)<br><sub>`_occupancy_mask_data_uri(grid: OccupancyGrid) -> str`</sub> | function | 模块级实现数据的内部辅助逻辑。 | `ImportError`, `np.zeros`, `Image.fromarray`, `BytesIO`, `image.save`, `decode`, `base64.b64encode`, `buffer.getvalue` |

## `saddlellm/SpatialWorldModel.py`

共 21 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SpatialPlanResult.to_dict`](../saddlellm/SpatialWorldModel.py#L38)<br><sub>`to_dict(self, include_grid: bool=False) -> Dict[str, Any]`</sub> | method | 把 `SpatialPlanResult` 转为可序列化字典。 | `enumerate`, `route.to_dict`, `list`, `self.grid.scale_to_source`, `routes.append`, `os.path.abspath`, `self.grid.to_dict`, `self.analysis.to_dict` |
| [`SpatialObservationEncoder.encode`](../saddlellm/SpatialWorldModel.py#L92)<br><sub>`encode(self, grid: OccupancyGrid, start: GridPoint, goal: GridPoint, observation_shape: Sequence[int], clearance: Optional[np.ndarray]=None) -> np.ndarray`</sub> | method | `SpatialObservationEncoder` 中编码`encode`的公开操作。 | `tuple`, `int`, `len`, `ValueError`, `grid.in_bounds`, `clearance_map`, `GridPathPlanner`, `max`, `math.atan2`, `astype` |
| [`SpatialObservationEncoder.describe`](../saddlellm/SpatialWorldModel.py#L143)<br><sub>`describe(self) -> Dict[str, Any]`</sub> | method | `SpatialObservationEncoder` 中描述`describe`的公开操作。 | `len`, `list` |
| [`SpatialBeliefMap.__init__`](../saddlellm/SpatialWorldModel.py#L154)<br><sub>`__init__(self, grid: OccupancyGrid) -> None`</sub> | method | 初始化 `SpatialBeliefMap` 实例及其运行依赖。 | `np.full_like`, `float` |
| [`SpatialBeliefMap.observe`](../saddlellm/SpatialWorldModel.py#L159)<br><sub>`observe(self, ground_truth: OccupancyGrid, center: GridPoint, radius: int) -> 'SpatialBeliefMap'`</sub> | method | `SpatialBeliefMap` 中实现`observe`的公开操作。 | `ValueError`, `max`, `min` |
| [`SpatialBeliefMap.as_grid`](../saddlellm/SpatialWorldModel.py#L179)<br><sub>`as_grid(self) -> OccupancyGrid`</sub> | method | `SpatialBeliefMap` 中实现占用栅格的公开操作。 | `OccupancyGrid`, `self.cells.copy` |
| [`SpatialBeliefMap.copy`](../saddlellm/SpatialWorldModel.py#L186)<br><sub>`copy(self) -> 'SpatialBeliefMap'`</sub> | method | `SpatialBeliefMap` 中实现`copy`的公开操作。 | `object.__new__`, `self.cells.copy` |
| [`SpatialTensorObservationEncoder.__init__`](../saddlellm/SpatialWorldModel.py#L218)<br><sub>`__init__(self, crop_size: int=16, sensor_radius: int=6) -> None`</sub> | method | 初始化 `SpatialTensorObservationEncoder` 实例及其运行依赖。 | `ValueError`, `int` |
| [`SpatialTensorObservationEncoder.observation_shape`](../saddlellm/SpatialWorldModel.py#L227)<br><sub>`observation_shape(self) -> Tuple[int, int, int]`</sub> | method | `SpatialTensorObservationEncoder` 中实现观测的公开操作。 | `len` |
| [`SpatialTensorObservationEncoder.new_belief`](../saddlellm/SpatialWorldModel.py#L230)<br><sub>`new_belief(self, grid: OccupancyGrid) -> SpatialBeliefMap`</sub> | method | `SpatialTensorObservationEncoder` 中实现`new_belief`的公开操作。 | `SpatialBeliefMap` |
| [`SpatialTensorObservationEncoder.encode`](../saddlellm/SpatialWorldModel.py#L233)<br><sub>`encode(self, grid: OccupancyGrid, current: GridPoint, goal: GridPoint, observation_shape: Optional[Sequence[int]]=None, belief: Optional[SpatialBeliefMap]=None, heading: float=0.0, dynamic: Optional[np.ndarray]=None, update_belief: bool=True) -> np.ndarray`</sub> | method | `SpatialTensorObservationEncoder` 中编码`encode`的公开操作。 | `tuple`, `int`, `ValueError`, `grid.in_bounds`, `belief.observe`, `np.asarray`, `np.full`, `np.zeros`, `range`, `float` |
| [`SpatialTensorObservationEncoder.describe`](../saddlellm/SpatialWorldModel.py#L311)<br><sub>`describe(self) -> Dict[str, Any]`</sub> | method | `SpatialTensorObservationEncoder` 中描述`describe`的公开操作。 | `list` |
| [`WorldModelRouteScorer.__init__`](../saddlellm/SpatialWorldModel.py#L323)<br><sub>`__init__(self, runtime: Any, action_encoder: Optional[Callable[[RouteCandidate], Any]]=None, discount: float=0.99, max_steps: int=64, geometric_cost_weight: float=0.05, risk_weight: float=1.0, learned_collision_weight: float=5.0) -> None`</sub> | method | 初始化 `WorldModelRouteScorer` 实例及其运行依赖。 | `ValueError`, `float`, `int` |
| [`WorldModelRouteScorer.score`](../saddlellm/SpatialWorldModel.py#L347)<br><sub>`score(self, state: RSSMState, routes: List[RouteCandidate]) -> None`</sub> | method | `WorldModelRouteScorer` 中评分`score`的公开操作。 | `self.action_encoder`, `self._continuous_actions`, `self.runtime.rollout`, `torch.pow`, `rewards.new_tensor`, `torch.arange`, `torch.cumprod`, `torch.cat`, `torch.ones`, `float` |
| [`WorldModelRouteScorer._continuous_actions`](../saddlellm/SpatialWorldModel.py#L396)<br><sub>`_continuous_actions(self, route: RouteCandidate) -> torch.Tensor`</sub> | method | `WorldModelRouteScorer` 中实现`continuous_actions`的内部辅助逻辑。 | `ValueError`, `_downsample_points`, `torch.zeros`, `max`, `len`, `enumerate`, `zip`, `float`, `math.hypot` |
| [`SpatialWorldModelCoordinator.__init__`](../saddlellm/SpatialWorldModel.py#L417)<br><sub>`__init__(self, extractor: Optional[TopDownMapExtractor]=None, planner: Optional[GridPathPlanner]=None, analyzer: Optional[Any]=None, route_scorer: Optional[WorldModelRouteScorer]=None, snap_radius: int=24, allow_perspective: bool=False) -> None`</sub> | method | 初始化 `SpatialWorldModelCoordinator` 实例及其运行依赖。 | `TopDownMapExtractor`, `GridPathPlanner`, `int`, `bool` |
| [`SpatialWorldModelCoordinator.plan_image`](../saddlellm/SpatialWorldModel.py#L433)<br><sub>`plan_image(self, image_path: str, start: Union[str, Sequence[float]], goal: Union[str, Sequence[float]], instruction: str='', route_count: Optional[int]=None, world_state: Optional[RSSMState]=None) -> SpatialPlanResult`</sub> | method | `SpatialWorldModelCoordinator` 中规划计划、图像的公开操作。 | `os.path.isfile`, `FileNotFoundError`, `self.extractor.extract`, `self.analyzer.analyze`, `SpatialAnalysis`, `ValueError`, `self.plan_grid` |
| [`SpatialWorldModelCoordinator.plan_grid`](../saddlellm/SpatialWorldModel.py#L473)<br><sub>`plan_grid(self, grid: OccupancyGrid, analysis: SpatialAnalysis, start: Union[str, Sequence[float]], goal: Union[str, Sequence[float]], instruction: str='', route_count: Optional[int]=None, world_state: Optional[RSSMState]=None, image_path: str='') -> SpatialPlanResult`</sub> | method | Plan on an already constructed world state. | `isinstance`, `TypeError`, `ValueError`, `self._resolve_point`, `self._snap_to_traversable`, `self.planner.plan`, `self.route_scorer.score`, `SpatialPlanResult` |
| [`SpatialWorldModelCoordinator._resolve_point`](../saddlellm/SpatialWorldModel.py#L524)<br><sub>`_resolve_point(self, value: Union[str, Sequence[float]], grid: OccupancyGrid, analysis: SpatialAnalysis) -> GridPoint`</sub> | method | `SpatialWorldModelCoordinator` 中解析`resolve_point`的内部辅助逻辑。 | `isinstance`, `grid.scale_from_source`, `lower`, `value.strip`, `entity.name.strip`, `ValueError`, `min`, `max`, `int`, `round` |
| [`SpatialWorldModelCoordinator._snap_to_traversable`](../saddlellm/SpatialWorldModel.py#L544)<br><sub>`_snap_to_traversable(self, grid: OccupancyGrid, point: GridPoint) -> GridPoint`</sub> | method | `SpatialWorldModelCoordinator` 中实现`snap_to_traversable`的内部辅助逻辑。 | `grid.traversable`, `deque`, `queue.popleft`, `grid.in_bounds`, `visited.add`, `queue.append`, `ValueError` |
| [`_downsample_points`](../saddlellm/SpatialWorldModel.py#L569)<br><sub>`_downsample_points(points: Sequence[GridPoint], maximum_points: int) -> List[GridPoint]`</sub> | function | 模块级实现`downsample_points`的内部辅助逻辑。 | `len`, `list`, `tolist`, `long`, `round`, `torch.linspace` |

## `saddlellm/SpatialWorldModelControl.py`

共 13 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SpatialCEMPlannerConfig.__post_init__`](../saddlellm/SpatialWorldModelControl.py#L45)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `SpatialCEMPlannerConfig` 创建后校验并规范化字段。 | `ValueError`, `getattr` |
| [`SpatialCEMPlan.action`](../saddlellm/SpatialWorldModelControl.py#L81)<br><sub>`action(self) -> torch.Tensor`</sub> | method | `SpatialCEMPlan` 中实现动作的公开操作。 | — |
| [`SpatialCEMPlan.to_dict`](../saddlellm/SpatialWorldModelControl.py#L84)<br><sub>`to_dict(self, include_occupancy: bool=True) -> Dict[str, Any]`</sub> | method | 把 `SpatialCEMPlan` 转为可序列化字典。 | `tolist`, `cpu`, `self.action.detach`, `self.actions.detach`, `float`, `list`, `int`, `self.predicted_rewards.detach`, `self.learned_collision_probability.detach`, `torch.softmax` |
| [`SpatialClosedLoopResult.to_dict`](../saddlellm/SpatialWorldModelControl.py#L117)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `SpatialClosedLoopResult` 转为可序列化字典。 | `list`, `self.last_plan.to_dict` |
| [`SpatialClosedLoopPlanner.__init__`](../saddlellm/SpatialWorldModelControl.py#L136)<br><sub>`__init__(self, runtime: Any, encoder: Optional[SpatialTensorObservationEncoder]=None, config: Optional[SpatialCEMPlannerConfig]=None) -> None`</sub> | method | 初始化 `SpatialClosedLoopPlanner` 实例及其运行依赖。 | `SpatialTensorObservationEncoder`, `SpatialCEMPlannerConfig`, `tuple`, `int`, `ValueError` |
| [`SpatialClosedLoopPlanner.plan`](../saddlellm/SpatialWorldModelControl.py#L156)<br><sub>`plan(self, state: RSSMState, grid: OccupancyGrid, current: GridPoint, goal: GridPoint) -> SpatialCEMPlan`</sub> | method | `SpatialClosedLoopPlanner` 中规划计划的公开操作。 | `torch.Generator`, `generator.manual_seed`, `torch.zeros`, `torch.ones_like`, `max`, `int`, `self._expert_seed`, `range`, `torch.randn`, `clamp` |
| [`SpatialClosedLoopPlanner.run_closed_loop`](../saddlellm/SpatialWorldModelControl.py#L254)<br><sub>`run_closed_loop(self, grid: OccupancyGrid, start: GridPoint, goal: GridPoint, max_steps: int=128) -> SpatialClosedLoopResult`</sub> | method | Execute one safe action per replan in the supplied grid simulator. | `ValueError`, `grid.traversable`, `self.encoder.new_belief`, `self.encoder.encode`, `range`, `self.runtime.filter`, `np.stack`, `self.runtime.encode_observation`, `self.plan`, `cpu` |
| [`SpatialClosedLoopPlanner._expert_seed`](../saddlellm/SpatialWorldModelControl.py#L331)<br><sub>`_expert_seed(self, grid: OccupancyGrid, current: GridPoint, goal: GridPoint, action_dim: int) -> torch.Tensor`</sub> | method | `SpatialClosedLoopPlanner` 中实现`expert_seed`的内部辅助逻辑。 | `torch.zeros`, `GridPathPlanner`, `SpatialPlannerConfig`, `planner.shortest_path`, `enumerate`, `zip`, `float` |
| [`SpatialClosedLoopPlanner._simulate_geometry`](../saddlellm/SpatialWorldModelControl.py#L357)<br><sub>`_simulate_geometry(self, grid: OccupancyGrid, current: GridPoint, actions: torch.Tensor) -> Tuple[List[GridPoint], int]`</sub> | method | `SpatialClosedLoopPlanner` 中模拟`simulate_geometry`的内部辅助逻辑。 | `self._apply_action`, `points.append`, `int` |
| [`SpatialClosedLoopPlanner._apply_action`](../saddlellm/SpatialWorldModelControl.py#L371)<br><sub>`_apply_action(self, grid: OccupancyGrid, current: GridPoint, action: Sequence[float]) -> Tuple[GridPoint, bool]`</sub> | method | `SpatialClosedLoopPlanner` 中应用动作的内部辅助逻辑。 | `_quantize_action`, `grid.traversable` |
| [`SpatialClosedLoopPlanner._geodesic_steps`](../saddlellm/SpatialWorldModelControl.py#L392)<br><sub>`_geodesic_steps(self, grid: OccupancyGrid, current: GridPoint, goal: GridPoint) -> float`</sub> | method | `SpatialClosedLoopPlanner` 中实现`geodesic_steps`的内部辅助逻辑。 | `GridPathPlanner`, `SpatialPlannerConfig`, `float`, `len`, `planner.shortest_path` |
| [`_quantize_action`](../saddlellm/SpatialWorldModelControl.py#L408)<br><sub>`_quantize_action(action: Sequence[float], diagonal: bool) -> Tuple[int, int]`</sub> | function | 模块级实现动作的内部辅助逻辑。 | `reshape`, `torch.as_tensor`, `values.numel`, `ValueError`, `float`, `math.hypot`, `abs` |
| [`_normalized_distance`](../saddlellm/SpatialWorldModelControl.py#L425)<br><sub>`_normalized_distance(grid: OccupancyGrid, point: GridPoint, goal: GridPoint) -> float`</sub> | function | 模块级实现`normalized_distance`的内部辅助逻辑。 | `max`, `math.hypot` |

## `saddlellm/SpatialWorldModelData.py`

共 13 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SpatialTrajectoryConfig.__post_init__`](../saddlellm/SpatialWorldModelData.py#L38)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `SpatialTrajectoryConfig` 创建后校验并规范化字段。 | `len`, `ValueError` |
| [`SpatialTrajectoryBuilder.__init__`](../saddlellm/SpatialWorldModelData.py#L60)<br><sub>`__init__(self, config: Optional[SpatialTrajectoryConfig]=None) -> None`</sub> | method | 初始化 `SpatialTrajectoryBuilder` 实例及其运行依赖。 | `SpatialTrajectoryConfig`, `SpatialObservationEncoder` |
| [`SpatialTrajectoryBuilder.build`](../saddlellm/SpatialWorldModelData.py#L64)<br><sub>`build(self, grid: OccupancyGrid, route: RouteCandidate, episode_id: str, metadata: Optional[Dict[str, Any]]=None) -> Dict[str, Any]`</sub> | method | `SpatialTrajectoryBuilder` 中构建`build`的公开操作。 | `_sample_route_points`, `len`, `ValueError`, `GridPathPlanner`, `planner.clearance_map`, `tolist`, `self.encoder.encode`, `enumerate`, `zip`, `float` |
| [`SpatialSequenceConfig.__post_init__`](../saddlellm/SpatialWorldModelData.py#L158)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `SpatialSequenceConfig` 创建后校验并规范化字段。 | `ValueError`, `getattr` |
| [`SpatialSequenceTrajectoryBuilder.__init__`](../saddlellm/SpatialWorldModelData.py#L187)<br><sub>`__init__(self, config: Optional[SpatialSequenceConfig]=None) -> None`</sub> | method | 初始化 `SpatialSequenceTrajectoryBuilder` 实例及其运行依赖。 | `SpatialSequenceConfig`, `SpatialTensorObservationEncoder` |
| [`SpatialSequenceTrajectoryBuilder.build`](../saddlellm/SpatialWorldModelData.py#L194)<br><sub>`build(self, grid: OccupancyGrid, route: RouteCandidate, episode_id: str, metadata: Optional[Dict[str, Any]]=None, rng: Optional[np.random.Generator]=None) -> Dict[str, Any]`</sub> | method | `SpatialSequenceTrajectoryBuilder` 中构建`build`的公开操作。 | `np.random.default_rng`, `_sample_route_points`, `len`, `ValueError`, `self.encoder.new_belief`, `clearance_map`, `GridPathPlanner`, `math.atan2`, `tolist`, `self.encoder.encode` |
| [`SpatialSequenceTrajectoryBuilder.build.append_transition`](../saddlellm/SpatialWorldModelData.py#L225)<br><sub>`append_transition(action_dx: float, action_dy: float, following: Tuple[int, int], collided: bool, terminal: bool) -> None`</sub> | nested function | `SpatialSequenceTrajectoryBuilder` 中追加`append_transition`的局部回调/辅助逻辑。 | `math.atan2`, `_wrap_angle`, `max`, `math.hypot`, `actions.append`, `ego_motions.append`, `_remaining_distance`, `min`, `float`, `rewards.append` |
| [`generate_spatial_world_model_dataset`](../saddlellm/SpatialWorldModelData.py#L336)<br><sub>`generate_spatial_world_model_dataset(image_path: str, output_path: str, episodes: int=64, routes_per_pair: int=2, seed: int=42, minimum_distance: float=0.25, map_config: Optional[MapExtractionConfig]=None, planner_config: Optional[SpatialPlannerConfig]=None, trajectory_config: Optional[SpatialTrajectoryConfig]=None) -> Dict[str, Any]`</sub> | function | Generate deterministic offline navigation episodes from one map image. | `ValueError`, `math.sqrt`, `Path`, `source.is_file`, `FileNotFoundError`, `extract`, `TopDownMapExtractor`, `MapExtractionConfig`, `str`, `replace` |
| [`generate_spatial_sequence_dataset`](../saddlellm/SpatialWorldModelData.py#L441)<br><sub>`generate_spatial_sequence_dataset(image_paths: Union[str, Sequence[str]], output_path: str, episodes: int=64, routes_per_pair: int=2, seed: int=42, minimum_distance: float=0.25, map_config: Optional[MapExtractionConfig]=None, planner_config: Optional[SpatialPlannerConfig]=None, sequence_config: Optional[SpatialSequenceConfig]=None) -> Dict[str, Any]`</sub> | function | Generate v2 temporal episodes across one or more top-down maps. | `isinstance`, `list`, `Path`, `ValueError`, `source.is_file`, `FileNotFoundError`, `math.sqrt`, `TopDownMapExtractor`, `MapExtractionConfig`, `extractor.extract` |
| [`_sample_route_points`](../saddlellm/SpatialWorldModelData.py#L558)<br><sub>`_sample_route_points(points: Sequence[Sequence[int]], maximum_points: int) -> List[tuple]`</sub> | function | 模块级采样路线的内部辅助逻辑。 | `int`, `len`, `astype`, `round`, `np.linspace`, `distinct.append` |
| [`_remaining_distance`](../saddlellm/SpatialWorldModelData.py#L572)<br><sub>`_remaining_distance(grid: OccupancyGrid, point: Sequence[int], goal: Sequence[int]) -> float`</sub> | function | 模块级实现`remaining_distance`的内部辅助逻辑。 | `float`, `max`, `math.hypot` |
| [`_adjacent_blocked_cell`](../saddlellm/SpatialWorldModelData.py#L580)<br><sub>`_adjacent_blocked_cell(grid: OccupancyGrid, point: Sequence[int], rng: np.random.Generator) -> Optional[Tuple[int, int]]`</sub> | function | 模块级实现`adjacent_blocked_cell`的内部辅助逻辑。 | `int`, `grid.in_bounds`, `grid.cell`, `candidates.append`, `rng.integers`, `len` |
| [`_wrap_angle`](../saddlellm/SpatialWorldModelData.py#L604)<br><sub>`_wrap_angle(value: float) -> float`</sub> | function | 模块级实现`wrap_angle`的内部辅助逻辑。 | `float` |

## `saddlellm/SpatialWorldModelEvaluation.py`

共 4 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`evaluate_spatial_world_model`](../saddlellm/SpatialWorldModelEvaluation.py#L15)<br><sub>`evaluate_spatial_world_model(checkpoint: str, data_path: str, device: str='auto', sequence_length: int=20, max_windows: Optional[int]=None, group_key: Optional[str]=None, group_value: Optional[str]=None) -> Dict[str, Any]`</sub> | function | Evaluate prior-only multi-step prediction on offline spatial sequences. | `ValueError`, `WorldModelRuntime.from_pretrained`, `int`, `getattr`, `load_world_model_trajectories`, `str`, `_nested_value`, `trajectory.get`, `WorldModelTrajectoryDataset`, `np.zeros` |
| [`_expected_calibration_error`](../saddlellm/SpatialWorldModelEvaluation.py#L180)<br><sub>`_expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int=10) -> float`</sub> | function | 模块级实现`expected_calibration_error`的内部辅助逻辑。 | `max`, `len`, `np.linspace`, `range`, `selected.any`, `float`, `mean`, `selected.sum`, `abs` |
| [`_occupancy_metrics`](../saddlellm/SpatialWorldModelEvaluation.py#L201)<br><sub>`_occupancy_metrics(confusion: np.ndarray) -> Dict[str, Any]`</sub> | function | 模块级实现指标的内部辅助逻辑。 | `astype`, `np.diag`, `confusion.sum`, `np.divide`, `np.zeros_like`, `float`, `intersection.sum`, `max`, `int`, `per_class_iou.mean` |
| [`_nested_value`](../saddlellm/SpatialWorldModelEvaluation.py#L218)<br><sub>`_nested_value(value: Any, path: str) -> Any`</sub> | function | 模块级实现`nested_value`的内部辅助逻辑。 | `split`, `str`, `isinstance` |

## `saddlellm/SwiftFullFineTuner.py`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`SwiftFullFineTuner.__init__`](../saddlellm/SwiftFullFineTuner.py#L15)<br><sub>`__init__(self, model_name: str='Qwen-7B', max_length: int=2048, use_flash_attn: bool=True, bf16: bool=True, device_map: str='auto')`</sub> | method | Initialize Swift for full fine-tuning. :param model_name: Model ID (e.g., "Qwen-7B", "Llama-3-8B") :param max_length: Max sequence length :param use_flash_attn: Enable FlashAttention-2 for speed :param bf16: Use bfloat16 precision (recomme… | `AutoModelForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained`, `model_name.lower`, `Swift.prepare_model` |
| [`SwiftFullFineTuner.fit`](../saddlellm/SwiftFullFineTuner.py#L51)<br><sub>`fit(self, train_dataset: Dataset, eval_dataset: Optional[Dataset]=None, epochs: int=3, batch_size: int=2, learning_rate: float=2e-05, output_dir: str='./output', logging_steps: int=10, save_strategy: str='steps', save_steps: int=500, gradient_accumulation_steps: int=4, deepspeed: Optional[str]=None) -> None`</sub> | method | Full fine-tuning with Swift. :param train_dataset: HF Dataset with "text" column :param deepspeed: Path to DeepSpeed config (for multi-GPU) | `train_dataset.map`, `eval_dataset.map`, `TrainingArguments`, `torch.cuda.is_bf16_supported`, `Trainer`, `self.trainer.train` |
| [`SwiftFullFineTuner.fit.tokenize_fn`](../saddlellm/SwiftFullFineTuner.py#L71)<br><sub>`tokenize_fn(examples: Dict) -> Dict`</sub> | nested function | `SwiftFullFineTuner` 中分词`tokenize_fn`的局部回调/辅助逻辑。 | `self.tokenizer` |
| [`SwiftFullFineTuner.predict`](../saddlellm/SwiftFullFineTuner.py#L112)<br><sub>`predict(self, text: str, max_new_tokens: int=100) -> str`</sub> | method | Generate text from input prompt. | `to`, `self.tokenizer`, `self.model.generate`, `self.tokenizer.decode` |
| [`SwiftFullFineTuner.save`](../saddlellm/SwiftFullFineTuner.py#L128)<br><sub>`save(self, path: str) -> None`</sub> | method | Save model and tokenizer. | `self.model.save_pretrained`, `self.tokenizer.save_pretrained` |
| [`SwiftFullFineTuner.load`](../saddlellm/SwiftFullFineTuner.py#L134)<br><sub>`load(cls, path: str, **kwargs) -> 'SwiftFullFineTuner'`</sub> | method | Load a saved model. | `cls` |
| [`load_imdb_dataset`](../saddlellm/SwiftFullFineTuner.py#L138)<br><sub>`load_imdb_dataset(dataset_path: str, sample_size: int=None)`</sub> | function | 加载IMDB数据集 | `os.path.join`, `os.listdir`, `open`, `texts.append`, `f.read`, `labels.append`, `Dataset.from_dict` |

## `saddlellm/TRLFullFineTuner.py`

共 6 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`generate_response`](../saddlellm/TRLFullFineTuner.py#L24)<br><sub>`generate_response(model, tokenizer, prompt, device='cuda', max_length=512)`</sub> | function | 生成模型响应 | `to`, `tokenizer`, `v.to`, `inputs.items`, `torch.cuda.amp.autocast`, `model.generate`, `tokenizer.decode` |
| [`ValidationCallback.__init__`](../saddlellm/TRLFullFineTuner.py#L43)<br><sub>`__init__(self, model, tokenizer, test_prompts, eval_steps)`</sub> | method | 初始化 `ValidationCallback` 实例及其运行依赖。 | — |
| [`ValidationCallback.on_step_end`](../saddlellm/TRLFullFineTuner.py#L49)<br><sub>`on_step_end(self, args, state, control, **kwargs)`</sub> | method | `ValidationCallback` 中实现`on_step_end`的公开操作。 | `print`, `self.model.eval`, `torch.no_grad`, `next`, `self.model.parameters`, `generate_response`, `len`, `self.model.train` |
| [`train_model`](../saddlellm/TRLFullFineTuner.py#L64)<br><sub>`train_model(model_path: str, dataset_path: str, output_path: str, learning_rate: float=2e-05, batch_size: int=2, num_epochs: int=3, save_steps: int=100, max_seq_length: int=512, gradient_accumulation_steps: int=4, warmup_steps: int=100, eval_steps: int=50, test_prompts: list=None, use_bf16: bool=True, use_gradient_checkpointing: bool=True, use_quantization: bool=False)`</sub> | function | 全参数微调主函数 | `torch.cuda.is_available`, `logging.info`, `load_dataset`, `ValueError`, `BitsAndBytesConfig`, `AutoModelForCausalLM.from_pretrained`, `next`, `model.parameters`, `AutoTokenizer.from_pretrained`, `DataCollatorForCompletionOnlyLM` |
| [`train_model.formatting_prompts_func`](../saddlellm/TRLFullFineTuner.py#L125)<br><sub>`formatting_prompts_func(example)`</sub> | nested function | 模块级实现`formatting_prompts_func`的局部回调/辅助逻辑。 | `range`, `len`, `tokenizer.encode`, `tokenizer.decode`, `texts.append` |
| [`parse_args`](../saddlellm/TRLFullFineTuner.py#L219)<br><sub>`parse_args()`</sub> | function | 命令行参数解析 | `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args` |

## `saddlellm/TextProcess.py`

共 25 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TextCleaner.__init__`](../saddlellm/TextProcess.py#L287)<br><sub>`__init__(self, config: Dict)`</sub> | method | 初始化 `TextCleaner` 实例及其运行依赖。 | `self._init_regex_patterns`, `config.get`, `SentenceTransformer` |
| [`TextCleaner._init_regex_patterns`](../saddlellm/TextProcess.py#L292)<br><sub>`_init_regex_patterns(self)`</sub> | method | 根据配置初始化正则表达式模式 | `self.config.get`, `re.escape`, `re.compile`, `join`, `map` |
| [`TextCleaner.clean_text`](../saddlellm/TextProcess.py#L313)<br><sub>`clean_text(self, text: str) -> Optional[str]`</sub> | method | 执行完整的文本清洗流程 | `isinstance`, `self.config.get`, `ftfy.fix_text`, `decode`, `text.encode`, `self.HTML_TAG_PATTERN.sub`, `html.unescape`, `self.JS_PATTERN.sub`, `self.CSS_PATTERN.sub`, `self.TABLE_PATTERN.sub` |
| [`TextCleaner._normalize_sentence_case`](../saddlellm/TextProcess.py#L454)<br><sub>`_normalize_sentence_case(self, text: str) -> str`</sub> | method | 将文本转换为句子格式 (首字母大写) | `re.split`, `upper`, `lower`, `join` |
| [`TextCleaner._normalize_numbers_to_digits`](../saddlellm/TextProcess.py#L460)<br><sub>`_normalize_numbers_to_digits(self, text: str) -> str`</sub> | method | 将中文数字转为阿拉伯数字 | `re.sub`, `cn_num_to_arabic`, `m.group` |
| [`TextCleaner._normalize_numbers_to_digits.cn_num_to_arabic`](../saddlellm/TextProcess.py#L463)<br><sub>`cn_num_to_arabic(cn_num)`</sub> | nested function | `TextCleaner` 中实现`cn_num_to_arabic`的局部回调/辅助逻辑。 | `str` |
| [`TextCleaner._extract_main_content`](../saddlellm/TextProcess.py#L493)<br><sub>`_extract_main_content(self, text: str) -> str`</sub> | method | 使用嵌入向量提取主要内容 | `self._segment_sentences`, `self.sentence_model.encode`, `fit`, `KMeans`, `np.argmax`, `np.bincount`, `range`, `len`, `join` |
| [`TextCleaner.detect_language`](../saddlellm/TextProcess.py#L512)<br><sub>`detect_language(text: str) -> str`</sub> | method | 检测文本语言 | `langdetect.detect` |
| [`TextCleaner.assess_quality`](../saddlellm/TextProcess.py#L520)<br><sub>`assess_quality(text: str) -> float`</sub> | method | 评估文本质量 (0-1) | `text.split`, `set`, `len`, `max`, `sum`, `c.isupper`, `min` |
| [`TextCleaner._expand_contractions`](../saddlellm/TextProcess.py#L543)<br><sub>`_expand_contractions(text: str) -> str`</sub> | method | 扩展英文缩写 | `contraction_map.items`, `re.sub` |
| [`TextCleaner._spell_out_numbers`](../saddlellm/TextProcess.py#L576)<br><sub>`_spell_out_numbers(text: str) -> str`</sub> | method | 将数字转为英文单词 (简单实现) | `re.sub`, `number_to_words`, `m.group` |
| [`TextCleaner._spell_out_numbers.number_to_words`](../saddlellm/TextProcess.py#L589)<br><sub>`number_to_words(num_str)`</sub> | nested function | `TextCleaner` 中实现`number_to_words`的局部回调/辅助逻辑。 | `int`, `str` |
| [`TextCleaner._segment_sentences`](../saddlellm/TextProcess.py#L607)<br><sub>`_segment_sentences(self, text: str) -> List[str]`</sub> | method | 分句处理 | `self.config.get`, `self.detect_language`, `sent_tokenize`, `list`, `jieba.cut`, `re.split`, `s.strip` |
| [`TextProcessor.__init__`](../saddlellm/TextProcess.py#L627)<br><sub>`__init__(self, config: Dict)`</sub> | method | 初始化 `TextProcessor` 实例及其运行依赖。 | `TextCleaner`, `set`, `config.get` |
| [`TextProcessor.process_file`](../saddlellm/TextProcess.py#L634)<br><sub>`process_file(self, input_path: str, output_path: str)`</sub> | method | 处理单个文件 | `suffix.lower`, `Path`, `open`, `json.loads`, `json.load`, `isinstance`, `line.strip`, `ValueError`, `logger.info`, `len` |
| [`TextProcessor._process_chunk`](../saddlellm/TextProcess.py#L700)<br><sub>`_process_chunk(self, chunk: List[Dict]) -> List[Dict]`</sub> | method | 处理数据块 | `self._process_item`, `results.append`, `logger.warning` |
| [`TextProcessor._process_item`](../saddlellm/TextProcess.py#L712)<br><sub>`_process_item(self, item: Dict) -> Optional[Dict]`</sub> | method | 处理单个文本项 | `item.get`, `self.cleaner.clean_text`, `result.items`, `self.config.get`, `self.cleaner._segment_sentences` |
| [`TextProcessor._remove_duplicates`](../saddlellm/TextProcess.py#L738)<br><sub>`_remove_duplicates(self, data: List[Dict]) -> List[Dict]`</sub> | method | 使用Simhash算法去重 | `item.get`, `Simhash`, `simhash.distance`, `self.duplicate_hashes.add`, `unique_data.append`, `logger.info`, `len` |
| [`TextProcessor._save_output`](../saddlellm/TextProcess.py#L764)<br><sub>`_save_output(self, data: List[Dict], output_path: str)`</sub> | method | 保存处理后的数据 | `os.makedirs`, `os.path.dirname`, `self.config.get`, `open`, `f.write`, `json.dumps`, `pd.DataFrame`, `pa.Table.from_pandas`, `pq.write_table`, `to_csv` |
| [`DataSourceHandler.handle_file`](../saddlellm/TextProcess.py#L807)<br><sub>`handle_file(input_path: str) -> List[Dict]`</sub> | method | 处理单个文件 | `TextProcessor._load_file` |
| [`DataSourceHandler.handle_directory`](../saddlellm/TextProcess.py#L812)<br><sub>`handle_directory(input_dir: str) -> List[Dict]`</sub> | method | 处理目录中的所有文件 | `glob`, `Path`, `file_path.is_file`, `data.extend`, `DataSourceHandler.handle_file`, `str`, `logger.warning` |
| [`DataSourceHandler.handle_api`](../saddlellm/TextProcess.py#L824)<br><sub>`handle_api(endpoint: str, params: Dict) -> List[Dict]`</sub> | method | 从API获取数据 | `requests.get`, `response.raise_for_status`, `response.json`, `logger.error` |
| [`DataSourceHandler.handle_database`](../saddlellm/TextProcess.py#L836)<br><sub>`handle_database(db_config: Dict) -> List[Dict]`</sub> | method | 从数据库获取数据 | `db_config.get`, `mysql.connector.connect`, `MongoClient`, `collection.find`, `json.loads`, `list`, `ValueError`, `conn.cursor`, `cursor.execute`, `cursor.fetchall` |
| [`load_config`](../saddlellm/TextProcess.py#L869)<br><sub>`load_config(config_path: str) -> Dict`</sub> | function | 加载配置文件 | `open`, `json.load`, `isinstance`, `k.strip`, `split` |
| [`main`](../saddlellm/TextProcess.py#L883)<br><sub>`main()`</sub> | function | 模块级实现`main`的公开操作。 | `argparse.ArgumentParser`, `parser.add_argument`, `os.cpu_count`, `parser.parse_args`, `load_config`, `TextProcessor`, `os.path.isdir`, `glob`, `Path`, `file.is_file` |

## `saddlellm/TokenizerLoader.py`

共 1 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`load_tokenizer_compatible`](../saddlellm/TokenizerLoader.py#L9)<br><sub>`load_tokenizer_compatible(model_path: str, *, trust_remote_code: bool=False, local_files_only: bool=False) -> Any`</sub> | function | Load with AutoTokenizer, falling back to a local tokenizer.json. | `AutoTokenizer.from_pretrained`, `expanduser`, `Path`, `tokenizer_file.is_file`, `config_path.is_file`, `json.loads`, `config_path.read_text`, `isinstance`, `config.get`, `value.get` |

## `saddlellm/TokenizerTrainer.py`

共 22 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TokenizerEvalResult.to_dict`](../saddlellm/TokenizerTrainer.py#L42)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `TokenizerEvalResult` 转为可序列化字典。 | — |
| [`TokenizerTrainer.__init__`](../saddlellm/TokenizerTrainer.py#L73)<br><sub>`__init__(self, vocab_size: int=32000, algorithm: str='bpe', min_frequency: int=2, special_tokens: Optional[List[str]]=None, max_token_length: int=128, byte_level: bool=True, unicode_normalizer: Optional[str]=None, chinese_char_coverage: float=0.995)`</sub> | method | 初始化 `TokenizerTrainer` 实例及其运行依赖。 | `TokenizerConfig` |
| [`TokenizerTrainer.fit`](../saddlellm/TokenizerTrainer.py#L96)<br><sub>`fit(self, corpus_files: Union[str, List[str]], file_format: str='auto', text_column: str='text', limit_gb: Optional[float]=None) -> 'TokenizerTrainer'`</sub> | method | 在语料上训练 tokenizer。 | `self._resolve_files`, `FileNotFoundError`, `logger.info`, `len`, `self._train_bpe`, `self._train_unigram`, `self._train_wordpiece`, `ValueError`, `self._tokenizer.get_vocab_size` |
| [`TokenizerTrainer._train_bpe`](../saddlellm/TokenizerTrainer.py#L130)<br><sub>`_train_bpe(self, files, file_format, text_column, limit_gb)`</sub> | method | 使用 HuggingFace tokenizers 库训练 BPE | `Tokenizer`, `models.BPE`, `pre_tokenizers.ByteLevel`, `decoders.ByteLevel`, `pre_tokenizers.Whitespace`, `normalizers.NFKC`, `normalizers.NFD`, `trainers.BpeTrainer`, `pre_tokenizers.ByteLevel.alphabet`, `tokenizer.train_from_iterator` |
| [`TokenizerTrainer._train_bpe.corpus_iterator`](../saddlellm/TokenizerTrainer.py#L152)<br><sub>`corpus_iterator()`</sub> | nested function | `TokenizerTrainer` 中实现`corpus_iterator`的局部回调/辅助逻辑。 | `float`, `self._detect_format`, `self._read_texts`, `len`, `text.strip`, `text.encode` |
| [`TokenizerTrainer._train_unigram`](../saddlellm/TokenizerTrainer.py#L177)<br><sub>`_train_unigram(self, files, file_format, text_column, limit_gb)`</sub> | method | 使用 SentencePiece 训练 Unigram 模型 | `ImportError`, `self._merge_corpus_to_temp`, `spm.SentencePieceTrainer.train`, `os.cpu_count`, `Tokenizer.from_file`, `os.path.exists`, `os.remove` |
| [`TokenizerTrainer._train_wordpiece`](../saddlellm/TokenizerTrainer.py#L209)<br><sub>`_train_wordpiece(self, files, file_format, text_column, limit_gb)`</sub> | method | 使用 HuggingFace tokenizers 训练 WordPiece | `Tokenizer`, `models.WordPiece`, `pre_tokenizers.BertPreTokenizer`, `decoders.WordPiece`, `trainers.WordPieceTrainer`, `tokenizer.train_from_iterator`, `corpus_iterator`, `self._tokenizer.enable_padding`, `self._tokenizer.enable_truncation` |
| [`TokenizerTrainer._train_wordpiece.corpus_iterator`](../saddlellm/TokenizerTrainer.py#L223)<br><sub>`corpus_iterator()`</sub> | nested function | `TokenizerTrainer` 中实现`corpus_iterator`的局部回调/辅助逻辑。 | `float`, `self._detect_format`, `self._read_texts`, `len`, `text.strip`, `text.encode` |
| [`TokenizerTrainer.save`](../saddlellm/TokenizerTrainer.py#L240)<br><sub>`save(self, path: str) -> str`</sub> | method | 保存 tokenizer 到目录 | `RuntimeError`, `os.makedirs`, `hasattr`, `self._tokenizer.save`, `os.path.join`, `open`, `json.dump`, `self._tokenizer.token_to_id`, `logger.info` |
| [`TokenizerTrainer.load`](../saddlellm/TokenizerTrainer.py#L287)<br><sub>`load(cls, path: str) -> 'TokenizerTrainer'`</sub> | method | 加载已保存的 tokenizer | `os.path.join`, `open`, `json.load`, `cls`, `config.get`, `os.path.exists`, `Tokenizer.from_file`, `FileNotFoundError` |
| [`TokenizerTrainer.encode`](../saddlellm/TokenizerTrainer.py#L310)<br><sub>`encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]`</sub> | method | 编码文本为 token IDs | `RuntimeError`, `isinstance`, `self._tokenizer.encode` |
| [`TokenizerTrainer.decode`](../saddlellm/TokenizerTrainer.py#L318)<br><sub>`decode(self, ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]`</sub> | method | 解码 token IDs 为文本 | `RuntimeError`, `isinstance`, `self._tokenizer.decode` |
| [`TokenizerTrainer.evaluate`](../saddlellm/TokenizerTrainer.py#L326)<br><sub>`evaluate(self, corpus_files: Optional[Union[str, List[str]]]=None, texts: Optional[Sequence[str]]=None, file_format: str='auto', text_column: str='text', max_samples: int=1000, domain_terms: Optional[Sequence[str]]=None, output_path: Optional[str]=None) -> TokenizerEvalResult`</sub> | method | Evaluate tokenizer quality before scratch pretraining. | `RuntimeError`, `hasattr`, `self._tokenizer.token_to_id`, `self._iter_eval_texts`, `text.strip`, `self._tokenizer.encode`, `list`, `getattr`, `len`, `sum` |
| [`TokenizerTrainer.get_hf_tokenizer`](../saddlellm/TokenizerTrainer.py#L388)<br><sub>`get_hf_tokenizer(self)`</sub> | method | 转换为 HuggingFace PreTrainedTokenizerFast (用于 transformers 训练) | `RuntimeError`, `os.path.join`, `os.path.dirname`, `hasattr`, `self._tokenizer.save`, `PreTrainedTokenizerFast`, `os.remove` |
| [`TokenizerTrainer.vocab_size`](../saddlellm/TokenizerTrainer.py#L411)<br><sub>`vocab_size(self) -> int`</sub> | method | `TokenizerTrainer` 中实现`vocab_size`的公开操作。 | `self._tokenizer.get_vocab_size` |
| [`TokenizerTrainer._iter_eval_texts`](../saddlellm/TokenizerTrainer.py#L418)<br><sub>`_iter_eval_texts(self, corpus_files: Optional[Union[str, List[str]]], texts: Optional[Sequence[str]], file_format: str, text_column: str) -> Iterator[str]`</sub> | method | `TokenizerTrainer` 中实现`iter_eval_texts`的内部辅助逻辑。 | `self._resolve_files`, `self._detect_format`, `self._read_texts` |
| [`TokenizerTrainer._evaluate_domain_terms`](../saddlellm/TokenizerTrainer.py#L435)<br><sub>`_evaluate_domain_terms(self, domain_terms: Sequence[str]) -> Dict[str, float]`</sub> | method | `TokenizerTrainer` 中评估`evaluate_domain_terms`的内部辅助逻辑。 | `self._tokenizer.encode`, `list`, `getattr`, `lengths.append`, `len`, `sum` |
| [`TokenizerTrainer._normalize_for_roundtrip`](../saddlellm/TokenizerTrainer.py#L453)<br><sub>`_normalize_for_roundtrip(self, text: str) -> str`</sub> | method | `TokenizerTrainer` 中规范化`normalize_for_roundtrip`的内部辅助逻辑。 | `join`, `split`, `strip` |
| [`TokenizerTrainer._resolve_files`](../saddlellm/TokenizerTrainer.py#L456)<br><sub>`_resolve_files(self, corpus_files) -> List[str]`</sub> | method | `TokenizerTrainer` 中解析`resolve_files`的内部辅助逻辑。 | `isinstance`, `glob.glob`, `resolved.extend`, `os.path.isfile`, `resolved.append`, `os.path.isdir`, `os.walk`, `f.endswith`, `os.path.join`, `sorted` |
| [`TokenizerTrainer._detect_format`](../saddlellm/TokenizerTrainer.py#L474)<br><sub>`_detect_format(self, fpath: str, hint: str) -> str`</sub> | method | `TokenizerTrainer` 中检测`detect_format`的内部辅助逻辑。 | `lower`, `os.path.splitext`, `get` |
| [`TokenizerTrainer._read_texts`](../saddlellm/TokenizerTrainer.py#L480)<br><sub>`_read_texts(self, fpath: str, fmt: str, text_column: str) -> Iterator[str]`</sub> | method | `TokenizerTrainer` 中读取`read_texts`的内部辅助逻辑。 | `open`, `_json.loads`, `obj.get`, `_json.load`, `isinstance`, `str`, `pq.read_table`, `table.to_batches`, `to_pylist`, `batch.column` |
| [`TokenizerTrainer._merge_corpus_to_temp`](../saddlellm/TokenizerTrainer.py#L507)<br><sub>`_merge_corpus_to_temp(self, files, fmt, text_column, limit_gb) -> str`</sub> | method | 将语料合并到临时文件 (SentencePiece 需要单个文件输入) | `tempfile.NamedTemporaryFile`, `float`, `self._detect_format`, `self._read_texts`, `len`, `text.strip`, `text.encode`, `tmp.write`, `tmp.close` |

## `saddlellm/TrainerUtils.py`

共 9 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`check_environment`](../saddlellm/TrainerUtils.py#L46)<br><sub>`check_environment() -> EnvCheckResult`</sub> | function | 全面环境自检。 | `torch.cuda.is_available`, `torch.cuda.device_count`, `torch.cuda.get_device_name`, `torch.cuda.get_device_properties`, `torch.cuda.memory_allocated`, `issues.append`, `psutil.virtual_memory`, `shutil.disk_usage`, `os.getcwd`, `recommendations.append` |
| [`TrainingDiagnostics.__init__`](../saddlellm/TrainerUtils.py#L195)<br><sub>`__init__(self)`</sub> | method | 初始化 `TrainingDiagnostics` 实例及其运行依赖。 | — |
| [`TrainingDiagnostics.check`](../saddlellm/TrainerUtils.py#L200)<br><sub>`check(self, step: int, loss: float, grad_norm: float=None, lr: float=None, mfu: float=None) -> Optional[str]`</sub> | method | 检查训练状态, 返回诊断信息 (如果有问题)。 | `self._loss_history.append`, `self._grad_norm_history.append`, `math.isnan`, `math.isinf`, `self._warnings.append`, `len`, `math.log`, `sum`, `max` |
| [`TrainingDiagnostics.get_summary`](../saddlellm/TrainerUtils.py#L289)<br><sub>`get_summary(self) -> Dict`</sub> | method | 获取训练诊断摘要。 | `len` |
| [`TrainingReportCard.generate`](../saddlellm/TrainerUtils.py#L321)<br><sub>`generate(model, training_config: Dict=None, metrics_history: List[Dict]=None, mfu_summary: Dict=None, eval_results: List=None, output_path: str=None) -> Dict`</sub> | method | 生成训练报告卡。 | `sum`, `p.numel`, `model.parameters`, `type`, `len`, `round`, `mfu_summary.get`, `improvements.append`, `print`, `training_config.get` |
| [`EarlyStopping.__init__`](../saddlellm/TrainerUtils.py#L410)<br><sub>`__init__(self, patience: int=5, min_delta: float=0.01, mode: str='min')`</sub> | method | 初始化 `EarlyStopping` 实例及其运行依赖。 | `float` |
| [`EarlyStopping.check`](../saddlellm/TrainerUtils.py#L418)<br><sub>`check(self, score: float) -> bool`</sub> | method | 检查是否应该早停。 | — |
| [`EarlyStopping.save_if_best`](../saddlellm/TrainerUtils.py#L437)<br><sub>`save_if_best(self, model, tokenizer, path: str, score: float)`</sub> | method | 如果是最佳模型就保存。 | `model.save_pretrained`, `tokenizer.save_pretrained`, `getattr` |
| [`auto_train_best_small_model`](../saddlellm/TrainerUtils.py#L451)<br><sub>`auto_train_best_small_model(model_size: str='300m', output_dir: str='./my_best_model', lang: str='zh', use_api_teacher: bool=False, api_teachers: List[str]=None)`</sub> | function | 一键训练最佳小模型。包含所有最佳实践。 | `print`, `check_environment`, `size_map.get`, `ModelRegistry.create_model`, `init_weights_llama_style`, `spec.human_params`, `DataCatalog.small_model_pack`, `AutoTokenizer.from_pretrained`, `DensePretrainConfig`, `os.path.join` |

## `saddlellm/TrainingConfigValidator.py`

共 5 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TrainingConfigValidation.to_dict`](../saddlellm/TrainingConfigValidator.py#L33)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `TrainingConfigValidation` 转为可序列化字典。 | `asdict` |
| [`TrainingConfigValidator.validate`](../saddlellm/TrainingConfigValidator.py#L41)<br><sub>`validate(cls, raw: Dict[str, Any], inspect_data: bool=True) -> TrainingConfigValidation`</sub> | method | `TrainingConfigValidator` 中校验`validate`的公开操作。 | `_json_safe`, `isinstance`, `TrainingOrchestrator._parse_config`, `TrainingConfigValidation`, `raw.get`, `list`, `issues.append`, `sorted`, `set`, `issues.extend` |
| [`TrainingConfigValidator._estimate`](../saddlellm/TrainingConfigValidator.py#L320)<br><sub>`_estimate(per_device_batch_size: int, gradient_accumulation_steps: int, num_gpus: int, max_seq_length: int, max_steps: int, epochs: Optional[int]) -> Dict[str, Any]`</sub> | method | `TrainingConfigValidator` 中估算`estimate`的内部辅助逻辑。 | `to_dict`, `TrainingPlanEstimator.estimate` |
| [`TrainingConfigValidator._inspect`](../saddlellm/TrainingConfigValidator.py#L340)<br><sub>`_inspect(data_inspections: Dict[str, Dict[str, Any]], issues: List[str], warnings: List[str], stage: str, path: str, task: str, inspect_data: bool) -> None`</sub> | method | `TrainingConfigValidator` 中检查`inspect`的内部辅助逻辑。 | `TrainingDataInspector.inspect_file`, `report.to_dict`, `issues.extend`, `warnings.extend` |
| [`validate_training_config`](../saddlellm/TrainingConfigValidator.py#L360)<br><sub>`validate_training_config(raw: Dict[str, Any], inspect_data: bool=True) -> Dict[str, Any]`</sub> | function | 模块级校验训练、配置的公开操作。 | `to_dict`, `TrainingConfigValidator.validate` |

## `saddlellm/TrainingDataInspector.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TrainingDataInspection.usable_ratio`](../saddlellm/TrainingDataInspector.py#L27)<br><sub>`usable_ratio(self) -> float`</sub> | method | `TrainingDataInspection` 中实现`usable_ratio`的公开操作。 | — |
| [`TrainingDataInspection.ready`](../saddlellm/TrainingDataInspector.py#L31)<br><sub>`ready(self) -> bool`</sub> | method | `TrainingDataInspection` 中实现`ready`的公开操作。 | — |
| [`TrainingDataInspection.severity`](../saddlellm/TrainingDataInspector.py#L35)<br><sub>`severity(self) -> str`</sub> | method | `TrainingDataInspection` 中实现`severity`的公开操作。 | — |
| [`TrainingDataInspection.to_dict`](../saddlellm/TrainingDataInspector.py#L42)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `TrainingDataInspection` 转为可序列化字典。 | `asdict` |
| [`TrainingDataInspector.inspect_file`](../saddlellm/TrainingDataInspector.py#L54)<br><sub>`inspect_file(cls, path: str, task: str='sft', max_records: int=256) -> TrainingDataInspection`</sub> | method | `TrainingDataInspector` 中检查`inspect_file`的公开操作。 | `TrainingDataInspection`, `report.errors.append`, `report.recommendations.append`, `os.path.exists`, `os.path.isfile`, `report.warnings.append`, `set`, `cls._iter_records`, `PostTrainingDataAdapter.detect_schema`, `report.detected_schemas.get` |
| [`TrainingDataInspector._schema_recommendation`](../saddlellm/TrainingDataInspector.py#L117)<br><sub>`_schema_recommendation(task: str) -> str`</sub> | method | `TrainingDataInspector` 中实现`schema_recommendation`的内部辅助逻辑。 | `lower` |
| [`TrainingDataInspector._record_signature`](../saddlellm/TrainingDataInspector.py#L128)<br><sub>`_record_signature(record: Dict) -> str`</sub> | method | `TrainingDataInspector` 中记录`record_signature`的内部辅助逻辑。 | `json.dumps` |
| [`TrainingDataInspector._lengths`](../saddlellm/TrainingDataInspector.py#L132)<br><sub>`_lengths(record: Dict) -> Tuple[int, int]`</sub> | method | `TrainingDataInspector` 中实现`lengths`的内部辅助逻辑。 | `record.get`, `len`, `str`, `max`, `join`, `m.get` |
| [`TrainingDataInspector._has_identical_preference_pair`](../saddlellm/TrainingDataInspector.py#L148)<br><sub>`_has_identical_preference_pair(record: Dict) -> bool`</sub> | method | `TrainingDataInspector` 中实现`has_identical_preference_pair`的内部辅助逻辑。 | `record.get`, `str` |
| [`TrainingDataInspector._iter_records`](../saddlellm/TrainingDataInspector.py#L152)<br><sub>`_iter_records(path: str, max_records: int) -> Iterable[Dict]`</sub> | method | `TrainingDataInspector` 中实现`iter_records`的内部辅助逻辑。 | `lower`, `os.path.splitext`, `open`, `line.strip`, `json.loads`, `json.load`, `isinstance`, `data.get`, `csv.DictReader`, `dict` |
| [`inspect_training_data`](../saddlellm/TrainingDataInspector.py#L189)<br><sub>`inspect_training_data(path: str, task: str='sft', max_records: int=256) -> Dict`</sub> | function | 模块级检查训练、数据的公开操作。 | `to_dict`, `TrainingDataInspector.inspect_file` |

## `saddlellm/TrainingFactory.py`

共 23 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`FactoryConfig.to_dict`](../saddlellm/TrainingFactory.py#L28)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `FactoryConfig` 转为可序列化字典。 | `asdict` |
| [`FactoryStatus.to_dict`](../saddlellm/TrainingFactory.py#L41)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `FactoryStatus` 转为可序列化字典。 | `asdict` |
| [`LLMTrainingFactory.__init__`](../saddlellm/TrainingFactory.py#L59)<br><sub>`__init__(self, config: Optional[FactoryConfig]=None)`</sub> | method | 初始化 `LLMTrainingFactory` 实例及其运行依赖。 | `FactoryConfig` |
| [`LLMTrainingFactory.for_domain`](../saddlellm/TrainingFactory.py#L63)<br><sub>`for_domain(cls, domain: str, root_dir: Optional[str]=None, base_model: str='Qwen/Qwen2.5-7B-Instruct', **kwargs) -> 'LLMTrainingFactory'`</sub> | method | `LLMTrainingFactory` 中实现`for_domain`的公开操作。 | `cls`, `FactoryConfig` |
| [`LLMTrainingFactory.create_workspace`](../saddlellm/TrainingFactory.py#L77)<br><sub>`create_workspace(self) -> Dict[str, str]`</sub> | method | `LLMTrainingFactory` 中创建`create_workspace`的公开操作。 | `os.makedirs`, `self.DIRS.items`, `os.path.join`, `self._save_json`, `self.config.to_dict`, `self.save_blueprint` |
| [`LLMTrainingFactory.blueprint`](../saddlellm/TrainingFactory.py#L88)<br><sub>`blueprint(self) -> Dict`</sub> | method | `LLMTrainingFactory` 中实现模型蓝图的公开操作。 | — |
| [`LLMTrainingFactory.save_blueprint`](../saddlellm/TrainingFactory.py#L122)<br><sub>`save_blueprint(self, path: Optional[str]=None) -> str`</sub> | method | `LLMTrainingFactory` 中保存模型蓝图的公开操作。 | `os.path.join`, `os.makedirs`, `os.path.dirname`, `self.blueprint`, `join`, `lines.extend`, `enumerate`, `open`, `f.write`, `self._save_json` |
| [`LLMTrainingFactory.plan`](../saddlellm/TrainingFactory.py#L143)<br><sub>`plan(self) -> Dict`</sub> | method | `LLMTrainingFactory` 中规划计划的公开操作。 | `DomainModelBuilder`, `builder.architecture_report`, `builder.training_strategy`, `builder.data_mix_plan`, `os.path.join`, `builder.eval_suite`, `self.backend_plan`, `self.blueprint` |
| [`LLMTrainingFactory.import_dataset_manifest`](../saddlellm/TrainingFactory.py#L164)<br><sub>`import_dataset_manifest(self, manifest_path: str, dataset_dir: Optional[str]=None, selected_names: Optional[Sequence[str]]=None, role: Optional[str]=None, save_path: Optional[str]=None) -> Dict`</sub> | method | Normalize a dataset registry into SaddleLLM source dictionaries. | `self.create_workspace`, `os.path.basename`, `DatasetManifest.from_llamafactory_json`, `DatasetManifest.load`, `os.path.join`, `manifest.save`, `manifest.to_dict`, `manifest.summary`, `manifest.to_saddle_sources`, `manifest.source_weights` |
| [`LLMTrainingFactory.backend_plan`](../saddlellm/TrainingFactory.py#L193)<br><sub>`backend_plan(self, model_params: int, num_gpus: Optional[int]=None, gpu_memory_gb: Optional[float]=None, seq_length: Optional[int]=None, is_moe: bool=False, stage: str='pretrain', save: bool=True, path: Optional[str]=None) -> Dict`</sub> | method | Create a backend and parallelism plan for a training stage. | `FactoryBackendPlanner.recommend_parallelism`, `plan.to_dict`, `FactoryBackendPlanner.to_training_orchestrator_distributed`, `FactoryBackendPlanner.to_megatron_style_args`, `FactoryBackendPlanner.to_colossal_plugin_spec`, `self.create_workspace`, `os.path.join`, `FactoryBackendPlanner.save_plan` |
| [`LLMTrainingFactory.model_blueprints`](../saddlellm/TrainingFactory.py#L230)<br><sub>`model_blueprints(self, save_path: Optional[str]=None) -> Dict`</sub> | method | Generate model-first architecture blueprints for this factory domain. | `ModelBlueprintLab.dense_gqa`, `ModelBlueprintLab.deepseek_style_moe`, `ModelBlueprintLab.minimax_style_long_context`, `ModelBlueprintLab.glm_style_reasoning`, `ModelBlueprintLab.compare`, `ModelBlueprintLab.save_comparison` |
| [`LLMTrainingFactory.create_model_experiments`](../saddlellm/TrainingFactory.py#L246)<br><sub>`create_model_experiments(self, data_sources: Optional[Sequence[Dict]]=None, dry_run: bool=True, **overrides) -> Dict`</sub> | method | Create controlled model-architecture experiment plans. | `self.create_workspace`, `ModelExperimentConfig`, `overrides.pop`, `os.path.join`, `list`, `run`, `ModelExperimentPlanner`, `bundle.to_dict` |
| [`LLMTrainingFactory.create_multimodal_plan`](../saddlellm/TrainingFactory.py#L270)<br><sub>`create_multimodal_plan(self, data_path: Optional[str]=None, image_root: Optional[str]=None, save: bool=True, **overrides) -> Dict`</sub> | method | Create a low-cost VLM/MLLM training plan. | `self.create_workspace`, `overrides.pop`, `os.path.join`, `ModelBlueprintLab.llava_style_vlm`, `TrainingRecipe`, `RecipeModelConfig`, `blueprint.estimate_total_params`, `RecipeDataConfig`, `RecipeMultimodalConfig`, `RecipeTrainingConfig` |
| [`LLMTrainingFactory.create_post_training_plan`](../saddlellm/TrainingFactory.py#L360)<br><sub>`create_post_training_plan(self, data_path: str, stage: str='sft', method: str='lora', preference_method: str='dpo', save: bool=True, **overrides) -> Dict`</sub> | method | Create a traditional text LLM post-training plan. | `self.create_workspace`, `stage.lower`, `ValueError`, `overrides.pop`, `os.path.join`, `to_dict`, `TrainingDataInspector.inspect_file`, `os.path.exists`, `PostTrainingDataAdapter.normalize_file`, `TrainingRecipe` |
| [`LLMTrainingFactory.create_preflight_plan`](../saddlellm/TrainingFactory.py#L455)<br><sub>`create_preflight_plan(self, data_path: str, stage: str='sft', method: str='lora', preference_method: str='dpo', save: bool=True, **overrides) -> Dict`</sub> | method | Create a no-training post-training preflight plan. | `overrides.pop`, `os.path.join`, `self.create_post_training_plan`, `bool`, `get`, `payload.get`, `list`, `self._estimate_from_orchestrator_config`, `os.makedirs`, `TrainingRecipe.from_dict` |
| [`LLMTrainingFactory._estimate_from_orchestrator_config`](../saddlellm/TrainingFactory.py#L498)<br><sub>`_estimate_from_orchestrator_config(self, config: Dict, stage: str) -> Dict`</sub> | method | `LLMTrainingFactory` 中估算配置的内部辅助逻辑。 | `config.get`, `stage_cfg.get`, `get`, `to_dict`, `TrainingPlanEstimator.estimate`, `training.get`, `distributed.get` |
| [`LLMTrainingFactory.create_mopd_plan`](../saddlellm/TrainingFactory.py#L524)<br><sub>`create_mopd_plan(self, prompts_path: str, teachers: Optional[Sequence[Dict]]=None, chain_sft: bool=True, save: bool=True, **overrides) -> Dict`</sub> | method | Create an on-policy multi-teacher distillation plan. | `self.create_workspace`, `overrides.pop`, `os.path.join`, `TrainingRecipe`, `RecipeModelConfig`, `RecipeMethodConfig`, `RecipeDataConfig`, `RecipeMOPDConfig`, `list`, `RecipeTrainingConfig` |
| [`LLMTrainingFactory.create_vla_plan`](../saddlellm/TrainingFactory.py#L611)<br><sub>`create_vla_plan(self, data_path: str, image_root: Optional[str]=None, save: bool=True, **overrides) -> Dict`</sub> | method | Create a vision-language-action behavior-cloning plan. | `self.create_workspace`, `overrides.pop`, `os.path.join`, `VLAActionSpace`, `os.path.exists`, `VLADataAdapter.normalize_file`, `TrainingRecipe`, `RecipeModelConfig`, `RecipeDataConfig`, `RecipeMultimodalConfig` |
| [`LLMTrainingFactory.save_plan`](../saddlellm/TrainingFactory.py#L766)<br><sub>`save_plan(self, path: Optional[str]=None) -> str`</sub> | method | `LLMTrainingFactory` 中保存计划的公开操作。 | `self.create_workspace`, `os.path.join`, `self._save_json`, `self.plan` |
| [`LLMTrainingFactory.create_pretrain_experiments`](../saddlellm/TrainingFactory.py#L771)<br><sub>`create_pretrain_experiments(self, corpus_sources: Optional[Sequence[Dict]]=None, sources_by_bucket: Optional[Dict[str, Sequence[Dict]]]=None, dry_run: bool=True, **overrides) -> Dict`</sub> | method | `LLMTrainingFactory` 中创建`create_pretrain_experiments`的公开操作。 | `self.create_workspace`, `PretrainExperimentConfig`, `overrides.pop`, `os.path.join`, `list`, `run`, `PretrainExperimentRunner`, `bundle.to_dict` |
| [`LLMTrainingFactory.report`](../saddlellm/TrainingFactory.py#L796)<br><sub>`report(self, output_dir: Optional[str]=None) -> Dict`</sub> | method | `LLMTrainingFactory` 中报告报告的公开操作。 | `PretrainReport.generate`, `os.path.join`, `os.makedirs`, `self._save_json`, `result.to_dict` |
| [`LLMTrainingFactory.status`](../saddlellm/TrainingFactory.py#L806)<br><sub>`status(self) -> Dict`</sub> | method | `LLMTrainingFactory` 中实现`status`的公开操作。 | `os.path.isdir`, `os.path.join`, `self.DIRS.items`, `os.path.exists`, `to_dict`, `FactoryStatus` |
| [`LLMTrainingFactory._save_json`](../saddlellm/TrainingFactory.py#L830)<br><sub>`_save_json(self, path: str, data: Dict) -> str`</sub> | method | `LLMTrainingFactory` 中保存`save_json`的内部辅助逻辑。 | `os.makedirs`, `os.path.dirname`, `open`, `json.dump` |

## `saddlellm/TrainingMonitor.py`

共 8 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TrainingMonitor.__init__`](../saddlellm/TrainingMonitor.py#L68)<br><sub>`__init__(self, model_params: int, config: Optional[MFUConfig]=None, num_gpus: int=1, seq_length: int=2048, gradient_accumulation_steps: int=1)`</sub> | method | 初始化 `TrainingMonitor` 实例及其运行依赖。 | `MFUConfig`, `GPU_PEAK_TFLOPS.get`, `deque`, `logger.info` |
| [`TrainingMonitor.on_step_start`](../saddlellm/TrainingMonitor.py#L113)<br><sub>`on_step_start(self)`</sub> | method | `TrainingMonitor` 中实现`on_step_start`的公开操作。 | `time.time` |
| [`TrainingMonitor.on_step_end`](../saddlellm/TrainingMonitor.py#L116)<br><sub>`on_step_end(self, tokens_processed: int)`</sub> | method | 训练一步结束,记录指标 | `time.time`, `max`, `self._tps_window.append`, `self._mfu_window.append`, `self._step_times.append`, `self._avg`, `logger.warning` |
| [`TrainingMonitor.log_status`](../saddlellm/TrainingMonitor.py#L148)<br><sub>`log_status(self, step: int) -> Dict`</sub> | method | 记录当前状态,返回指标字典 | `self._avg`, `pynvml.nvmlInit`, `pynvml.nvmlDeviceGetHandleByIndex`, `pynvml.nvmlDeviceGetUtilizationRates`, `pynvml.nvmlDeviceGetMemoryInfo`, `round`, `max`, `parts.append`, `logger.info`, `join` |
| [`TrainingMonitor.summary`](../saddlellm/TrainingMonitor.py#L194)<br><sub>`summary(self) -> Dict`</sub> | method | 训练结束,生成汇总报告 | `self._avg`, `max`, `round`, `logger.info` |
| [`TrainingMonitor.estimate_completion`](../saddlellm/TrainingMonitor.py#L247)<br><sub>`estimate_completion(self, total_steps: int, current_step: int) -> Dict`</sub> | method | 估算训练完成时间 | `self._avg`, `round` |
| [`TrainingMonitor._avg`](../saddlellm/TrainingMonitor.py#L269)<br><sub>`_avg(self, window: deque) -> float`</sub> | method | `TrainingMonitor` 中实现`avg`的内部辅助逻辑。 | `sum`, `len` |
| [`TrainingMonitor.get_latest_metrics`](../saddlellm/TrainingMonitor.py#L274)<br><sub>`get_latest_metrics(self) -> Dict`</sub> | method | 获取最新指标 (供外部回调使用) | `round`, `self._avg` |

## `saddlellm/TrainingOrchestrator.py`

共 42 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TrainingOrchestrator.__init__`](../saddlellm/TrainingOrchestrator.py#L464)<br><sub>`__init__(self, config: TrainingConfig)`</sub> | method | 初始化 `TrainingOrchestrator` 实例及其运行依赖。 | `self._validate_config`, `current_distributed_runtime`, `os.makedirs`, `self._setup_logging` |
| [`TrainingOrchestrator._validate_config`](../saddlellm/TrainingOrchestrator.py#L481)<br><sub>`_validate_config(self)`</sub> | method | `TrainingOrchestrator` 中校验配置的内部辅助逻辑。 | `ValueError`, `sorted`, `self.config.stages.index`, `self._has_upstream_media_cache`, `self.config.world_model.data.get`, `max`, `os.environ.get`, `validate_distributed_runtime`, `set`, `join` |
| [`TrainingOrchestrator._has_upstream_media_cache`](../saddlellm/TrainingOrchestrator.py#L652)<br><sub>`_has_upstream_media_cache(self, modality: str, consumer: str) -> bool`</sub> | method | `TrainingOrchestrator` 中实现缓存的内部辅助逻辑。 | `stages.index`, `any`, `isinstance`, `lower`, `str`, `job.get` |
| [`TrainingOrchestrator._resolve_upstream_media_cache`](../saddlellm/TrainingOrchestrator.py#L668)<br><sub>`_resolve_upstream_media_cache(self, modality: str) -> str`</sub> | method | `TrainingOrchestrator` 中解析缓存的内部辅助逻辑。 | `self._stage_results.get`, `get`, `result.get`, `ValueError` |
| [`TrainingOrchestrator._validate_architecture_support`](../saddlellm/TrainingOrchestrator.py#L677)<br><sub>`_validate_architecture_support(self)`</sub> | method | `TrainingOrchestrator` 中校验`validate_architecture_support`的内部辅助逻辑。 | `MODEL_SPECS.get`, `ArchitectureRegistry.require`, `ArchitectureRegistry.detect_from_model_name`, `support.supports`, `ValueError`, `stage_capabilities.items` |
| [`TrainingOrchestrator._default_blueprint_for_spec`](../saddlellm/TrainingOrchestrator.py#L706)<br><sub>`_default_blueprint_for_spec(self, spec)`</sub> | method | `TrainingOrchestrator` 中实现模型蓝图的内部辅助逻辑。 | `ModelBlueprint`, `AttentionBlueprint`, `FFNBlueprint`, `ObjectiveBlueprint` |
| [`TrainingOrchestrator._resolve_model_blueprint`](../saddlellm/TrainingOrchestrator.py#L753)<br><sub>`_resolve_model_blueprint(self, spec=None)`</sub> | method | `TrainingOrchestrator` 中解析模型、模型蓝图的内部辅助逻辑。 | `MODEL_SPECS.get`, `raw.get`, `required.add`, `sorted`, `ValueError`, `join`, `self._default_blueprint_for_spec`, `ModelBlueprint.from_dict` |
| [`TrainingOrchestrator._create_saddle_model`](../saddlellm/TrainingOrchestrator.py#L780)<br><sub>`_create_saddle_model(self, spec)`</sub> | method | `TrainingOrchestrator` 中创建模型的内部辅助逻辑。 | `KeyError`, `logger.info`, `ModelRegistry.create_saddle_model`, `self._resolve_model_blueprint`, `blueprint.build_model` |
| [`TrainingOrchestrator._resolve_exact_resume_checkpoint`](../saddlellm/TrainingOrchestrator.py#L795)<br><sub>`_resolve_exact_resume_checkpoint(self) -> Optional[str]`</sub> | method | Resolve and validate an exact Trainer checkpoint for native resume. | `os.path.join`, `os.path.isdir`, `get_last_checkpoint`, `FileNotFoundError`, `isinstance`, `os.path.abspath`, `os.fspath`, `TypeError`, `ValueError`, `sorted` |
| [`TrainingOrchestrator.from_yaml`](../saddlellm/TrainingOrchestrator.py#L864)<br><sub>`from_yaml(cls, path: str) -> 'TrainingOrchestrator'`</sub> | method | Create an orchestrator from a YAML config file. | `open`, `yaml.safe_load`, `cls._parse_config`, `cls` |
| [`TrainingOrchestrator.from_dict`](../saddlellm/TrainingOrchestrator.py#L874)<br><sub>`from_dict(cls, data: Dict) -> 'TrainingOrchestrator'`</sub> | method | 从字典解析并创建 `TrainingOrchestrator`。 | `cls._parse_config`, `cls` |
| [`TrainingOrchestrator.run`](../saddlellm/TrainingOrchestrator.py#L878)<br><sub>`run(self)`</sub> | method | Run all configured training stages. | `validate_distributed_runtime`, `logger.info`, `join`, `time.time`, `self.run_stage`, `str`, `logger.error`, `self._print_summary`, `self._close_logging_handlers` |
| [`TrainingOrchestrator.run_operator`](../saddlellm/TrainingOrchestrator.py#L923)<br><sub>`run_operator(cls, label: str, config: Optional[Dict[str, Any]], upstream_result: Optional[Dict[str, Any]], work_dir: str) -> Dict[str, Any]`</sub> | method | Run one visual operator through the normal stage scheduler. | `os.path.abspath`, `TrainingConfig`, `label.lower`, `LoggingConfig`, `EvalConfig`, `OperatorStageConfig`, `dict`, `cls`, `orchestrator.run`, `orchestrator._close_logging_handlers` |
| [`TrainingOrchestrator.run_stage`](../saddlellm/TrainingOrchestrator.py#L966)<br><sub>`run_stage(self, stage: str)`</sub> | method | Run a single training stage. | `getattr`, `ValueError`, `logger.info`, `stage.upper`, `self._run_tokenizer_stage`, `self._run_pretrain_stage`, `self._run_sft_stage`, `self._run_preference_stage`, `self._run_rlhf_stage`, `self._run_mopd_stage` |
| [`TrainingOrchestrator._run_media_cache_stage`](../saddlellm/TrainingOrchestrator.py#L1020)<br><sub>`_run_media_cache_stage(self)`</sub> | method | Build resumable raw-media caches for downstream generation stages. | `set`, `enumerate`, `isinstance`, `TypeError`, `dict`, `str`, `job.get`, `os.path.isabs`, `os.path.join`, `MediaCacheBuildConfig` |
| [`TrainingOrchestrator._run_operator_stage`](../saddlellm/TrainingOrchestrator.py#L1074)<br><sub>`_run_operator_stage(self)`</sub> | method | Execute a visual-flow LLM operator as a first-class stage. | `run_agent_operator`, `run_large_model_operator`, `str`, `resolve`, `Path`, `error_dir.mkdir`, `error_path.write_text`, `json.dumps`, `append`, `RuntimeError` |
| [`TrainingOrchestrator._run_tokenizer_stage`](../saddlellm/TrainingOrchestrator.py#L1137)<br><sub>`_run_tokenizer_stage(self)`</sub> | method | Stage 1: train tokenizer. | `logger.info`, `self._load_tokenizer`, `TokenizerTrainer`, `corpus_files.append`, `trainer.fit`, `trainer.save`, `trainer.evaluate`, `os.path.join`, `trainer.get_hf_tokenizer`, `eval_result.to_dict` |
| [`TrainingOrchestrator._run_pretrain_stage`](../saddlellm/TrainingOrchestrator.py#L1183)<br><sub>`_run_pretrain_stage(self)`</sub> | method | Stage 2: pretrain model. | `self._resolve_exact_resume_checkpoint`, `MODEL_SPECS.get`, `KeyError`, `list`, `MODEL_SPECS.keys`, `logger.info`, `spec.human_params`, `spec.human_tokens`, `self._resolve_model_blueprint`, `blueprint.analyze` |
| [`TrainingOrchestrator._inspect_post_training_data`](../saddlellm/TrainingOrchestrator.py#L1399)<br><sub>`_inspect_post_training_data(self, path: str, task: str) -> Dict`</sub> | method | `TrainingOrchestrator` 中检查训练、数据的内部辅助逻辑。 | `TrainingDataInspector.inspect_file`, `self._close_logging_handlers`, `ValueError`, `join`, `report.to_dict` |
| [`TrainingOrchestrator._safe_dataclass_dict`](../saddlellm/TrainingOrchestrator.py#L1408)<br><sub>`_safe_dataclass_dict(self, value: Any) -> Dict`</sub> | method | `TrainingOrchestrator` 中实现`safe_dataclass_dict`的内部辅助逻辑。 | `is_dataclass`, `asdict`, `isinstance`, `dict` |
| [`TrainingOrchestrator._write_stage_plan`](../saddlellm/TrainingOrchestrator.py#L1415)<br><sub>`_write_stage_plan(self, stage: str, payload: Dict) -> Dict`</sub> | method | `TrainingOrchestrator` 中写入训练阶段、计划的内部辅助逻辑。 | `os.path.join`, `os.makedirs`, `open`, `json.dump` |
| [`TrainingOrchestrator._estimate_stage_training`](../saddlellm/TrainingOrchestrator.py#L1423)<br><sub>`_estimate_stage_training(self, per_device_batch_size: int, gradient_accumulation_steps: int, max_seq_length: int, epochs: Optional[int]=None) -> Dict`</sub> | method | `TrainingOrchestrator` 中估算训练阶段、训练的内部辅助逻辑。 | `to_dict`, `TrainingPlanEstimator.estimate` |
| [`TrainingOrchestrator._run_sft_stage`](../saddlellm/TrainingOrchestrator.py#L1441)<br><sub>`_run_sft_stage(self)`</sub> | method | Stage 3: supervised fine-tuning. | `logger.info`, `get`, `self._stage_results.get`, `ValueError`, `self._inspect_post_training_data`, `os.path.join`, `self._write_stage_plan`, `self._estimate_stage_training`, `self._safe_dataclass_dict`, `train_native_sft` |
| [`TrainingOrchestrator._run_preference_stage`](../saddlellm/TrainingOrchestrator.py#L1560)<br><sub>`_run_preference_stage(self)`</sub> | method | Stage 4: preference training with DPO, ORPO, or KTO. | `logger.info`, `get`, `self._stage_results.get`, `ValueError`, `os.path.join`, `self._inspect_post_training_data`, `self._write_stage_plan`, `self._estimate_stage_training`, `self._safe_dataclass_dict`, `train_native_dpo` |
| [`TrainingOrchestrator._run_rlhf_stage`](../saddlellm/TrainingOrchestrator.py#L1668)<br><sub>`_run_rlhf_stage(self)`</sub> | method | Stage 4: RLHF alignment. | `logger.info`, `get`, `self._stage_results.get`, `ValueError`, `os.path.join`, `self._inspect_post_training_data`, `self._write_stage_plan`, `self._estimate_stage_training`, `self._safe_dataclass_dict`, `train_native_dpo` |
| [`TrainingOrchestrator._run_multimodal_stage`](../saddlellm/TrainingOrchestrator.py#L1778)<br><sub>`_run_multimodal_stage(self)`</sub> | method | Materialize multimodal/VLM training plans and normalized data. | `os.path.join`, `os.makedirs`, `os.path.exists`, `MultimodalDataAdapter.normalize_file`, `report.to_dict`, `open`, `json.dump`, `plan.get` |
| [`TrainingOrchestrator._run_image_generation_stage`](../saddlellm/TrainingOrchestrator.py#L1829)<br><sub>`_run_image_generation_stage(self)`</sub> | method | Train a text-conditioned flow model from cached image latents. | `get`, `self._stage_results.get`, `self._resolve_upstream_media_cache`, `CachedLatentDataset`, `getattr`, `cache_manifest.get`, `dataset.infer_model_config`, `dimensions.items`, `int`, `ValueError` |
| [`TrainingOrchestrator._run_music_generation_stage`](../saddlellm/TrainingOrchestrator.py#L1933)<br><sub>`_run_music_generation_stage(self)`</sub> | method | Train a text-conditioned AR model from cached audio codec tokens. | `get`, `self._stage_results.get`, `self._resolve_upstream_media_cache`, `CachedMusicCodeDataset`, `getattr`, `cache_manifest.get`, `codec_fingerprint.get`, `dataset.infer_model_config`, `int`, `ValueError` |
| [`TrainingOrchestrator._run_video_generation_stage`](../saddlellm/TrainingOrchestrator.py#L2067)<br><sub>`_run_video_generation_stage(self)`</sub> | method | Train a text-conditioned flow model from cached video latents. | `get`, `self._stage_results.get`, `self._resolve_upstream_media_cache`, `CachedVideoLatentDataset`, `getattr`, `cache_manifest.get`, `dataset.infer_model_config`, `dimensions.items`, `int`, `ValueError` |
| [`TrainingOrchestrator._run_world_model_stage`](../saddlellm/TrainingOrchestrator.py#L2176)<br><sub>`_run_world_model_stage(self)`</sub> | method | Run the native action-conditioned world-model trainer. | `dict`, `training.get`, `os.path.isabs`, `os.path.join`, `train_world_model_from_config`, `self._write_stage_plan`, `os.path.abspath`, `result.get` |
| [`TrainingOrchestrator._run_vla_stage`](../saddlellm/TrainingOrchestrator.py#L2220)<br><sub>`_run_vla_stage(self)`</sub> | method | Materialize VLA behavior-cloning plans and normalized robot data. | `os.path.join`, `os.makedirs`, `VLAActionSpace`, `VLATrainingPlanner.create_plan`, `plan.update`, `os.path.exists`, `VLADataAdapter.normalize_file`, `report.to_dict`, `plan.get`, `train_vla_sft` |
| [`TrainingOrchestrator._run_mopd_stage`](../saddlellm/TrainingOrchestrator.py#L2326)<br><sub>`_run_mopd_stage(self)`</sub> | method | Materialize or collect MOPD-style on-policy distillation data. | `os.path.join`, `os.makedirs`, `self._load_mopd_prompts`, `self._ensure_policy_model_loaded`, `OnPolicyTeacherSpec`, `t.get`, `float`, `self._build_mopd_teacher`, `enumerate`, `MOPDConfig` |
| [`TrainingOrchestrator._ensure_policy_model_loaded`](../saddlellm/TrainingOrchestrator.py#L2388)<br><sub>`_ensure_policy_model_loaded(self)`</sub> | method | Load the current policy model/tokenizer for rollout-based stages. | `get`, `self._stage_results.get`, `ValueError`, `self._load_tokenizer`, `torch.cuda.is_available`, `torch.cuda.is_bf16_supported`, `AutoModelForCausalLM.from_pretrained`, `self._model.to` |
| [`TrainingOrchestrator._build_mopd_teacher`](../saddlellm/TrainingOrchestrator.py#L2414)<br><sub>`_build_mopd_teacher(self, raw: Dict)`</sub> | method | `TrainingOrchestrator` 中构建`build_mopd_teacher`的内部辅助逻辑。 | `raw.get`, `os.environ.get`, `TeacherInterface.from_openai`, `TeacherInterface.from_anthropic`, `TeacherInterface.from_deepseek`, `TeacherInterface`, `ValueError` |
| [`TrainingOrchestrator._load_mopd_prompts`](../saddlellm/TrainingOrchestrator.py#L2433)<br><sub>`_load_mopd_prompts(path: str) -> List[str]`</sub> | method | `TrainingOrchestrator` 中加载`load_mopd_prompts`的内部辅助逻辑。 | `os.path.exists`, `lower`, `os.path.splitext`, `open`, `line.strip`, `json.loads`, `row.get`, `prompts.append`, `str`, `json.load` |
| [`TrainingOrchestrator._run_eval_stage`](../saddlellm/TrainingOrchestrator.py#L2463)<br><sub>`_run_eval_stage(self)`</sub> | method | Stage 5: evaluate model. | `logger.info`, `get`, `self._stage_results.get`, `ValueError`, `Evaluator`, `evaluator.evaluate`, `results.get`, `RuntimeError`, `str`, `json.dumps` |
| [`TrainingOrchestrator._run_export_stage`](../saddlellm/TrainingOrchestrator.py#L2527)<br><sub>`_run_export_stage(self)`</sub> | method | Package the latest trained checkpoint as a deployable HF release. | `get`, `self._stage_results.get`, `ValueError`, `isinstance`, `eval_result.get`, `bool`, `gate_payload.get`, `RuntimeError`, `os.path.join`, `os.path.abspath` |
| [`TrainingOrchestrator._load_tokenizer`](../saddlellm/TrainingOrchestrator.py#L2603)<br><sub>`_load_tokenizer(self)`</sub> | method | Load or reuse a tokenizer. | `load_tokenizer_compatible`, `os.path.exists`, `logger.warning` |
| [`TrainingOrchestrator._setup_logging`](../saddlellm/TrainingOrchestrator.py#L2628)<br><sub>`_setup_logging(self)`</sub> | method | `TrainingOrchestrator` 中实现`setup_logging`的内部辅助逻辑。 | `getattr`, `self.config.logging.log_level.upper`, `logging.Formatter`, `logging.getLogger`, `root.setLevel`, `os.path.join`, `list`, `root.removeHandler`, `handler.close`, `logging.StreamHandler` |
| [`TrainingOrchestrator._close_logging_handlers`](../saddlellm/TrainingOrchestrator.py#L2650)<br><sub>`_close_logging_handlers(self)`</sub> | method | `TrainingOrchestrator` 中关闭`close_logging_handlers`的内部辅助逻辑。 | `logging.getLogger`, `list`, `getattr`, `root.removeHandler`, `handler.close` |
| [`TrainingOrchestrator._print_summary`](../saddlellm/TrainingOrchestrator.py#L2657)<br><sub>`_print_summary(self, elapsed_seconds: Optional[float]=None)`</sub> | method | `TrainingOrchestrator` 中实现`print_summary`的内部辅助逻辑。 | `getattr`, `list`, `self._stage_results.keys`, `os.path.join`, `open`, `json.dump`, `logger.info`, `PretrainReport.generate`, `logger.warning` |
| [`TrainingOrchestrator._parse_config`](../saddlellm/TrainingOrchestrator.py#L2690)<br><sub>`_parse_config(raw: Dict) -> TrainingConfig`</sub> | method | Parse a YAML/JSON dictionary into TrainingConfig. | `raw.get`, `isinstance`, `model_raw.get`, `ValueError`, `get`, `data_sources.append`, `DataSourceConfig`, `s.get`, `TokenizerTrainingConfig`, `tok_raw.get` |

## `saddlellm/TrainingPlanEstimator.py`

共 3 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TrainingPlanEstimate.to_dict`](../saddlellm/TrainingPlanEstimator.py#L19)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `TrainingPlanEstimate` 转为可序列化字典。 | `asdict` |
| [`TrainingPlanEstimator.estimate`](../saddlellm/TrainingPlanEstimator.py#L25)<br><sub>`estimate(per_device_batch_size: int, gradient_accumulation_steps: int, num_gpus: int, max_seq_length: int, max_steps: int, epochs: Optional[int]=None) -> TrainingPlanEstimate`</sub> | method | `TrainingPlanEstimator` 中估算`estimate`的公开操作。 | `max`, `TrainingPlanEstimate` |
| [`estimate_training_plan`](../saddlellm/TrainingPlanEstimator.py#L54)<br><sub>`estimate_training_plan(per_device_batch_size: int, gradient_accumulation_steps: int, num_gpus: int, max_seq_length: int, max_steps: int, epochs: Optional[int]=None) -> Dict`</sub> | function | 模块级估算训练、计划的公开操作。 | `to_dict`, `TrainingPlanEstimator.estimate` |

## `saddlellm/TrainingRecipe.py`

共 20 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TrainingRecipe.to_dict`](../saddlellm/TrainingRecipe.py#L176)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `TrainingRecipe` 转为可序列化字典。 | `asdict` |
| [`TrainingRecipe.from_dict`](../saddlellm/TrainingRecipe.py#L180)<br><sub>`from_dict(cls, raw: Dict) -> 'TrainingRecipe'`</sub> | method | 从字典解析并创建 `TrainingRecipe`。 | `cls`, `raw.get`, `RecipeModelConfig`, `RecipeMethodConfig`, `RecipeDataConfig`, `RecipeMultimodalConfig`, `RecipeMOPDConfig`, `cls._parse_vla_config`, `RecipeBackendConfig`, `RecipeTrainingConfig` |
| [`TrainingRecipe._parse_vla_config`](../saddlellm/TrainingRecipe.py#L199)<br><sub>`_parse_vla_config(raw: Dict) -> RecipeVLAConfig`</sub> | method | `TrainingRecipe` 中解析配置的内部辅助逻辑。 | `dict`, `data.pop`, `isinstance`, `aliases.items`, `RecipeVLAConfig` |
| [`TrainingRecipe.load`](../saddlellm/TrainingRecipe.py#L222)<br><sub>`load(cls, path: str) -> 'TrainingRecipe'`</sub> | method | `TrainingRecipe` 中加载`load`的公开操作。 | `open`, `endswith`, `path.lower`, `cls.from_dict`, `yaml.safe_load`, `json.load` |
| [`TrainingRecipe.save`](../saddlellm/TrainingRecipe.py#L229)<br><sub>`save(self, path: str) -> str`</sub> | method | `TrainingRecipe` 中保存`save`的公开操作。 | `os.makedirs`, `os.path.dirname`, `self.to_dict`, `open`, `endswith`, `path.lower`, `yaml.safe_dump`, `json.dump` |
| [`TrainingRecipe.template`](../saddlellm/TrainingRecipe.py#L241)<br><sub>`template(cls, stage: str='sft', domain: str='research', output_dir: str='./outputs', base_model: str='Qwen/Qwen2.5-7B-Instruct') -> 'TrainingRecipe'`</sub> | method | `TrainingRecipe` 中实现`template`的公开操作。 | `RecipeModelConfig`, `get`, `cls`, `RecipeDataConfig`, `RecipeLoggingConfig` |
| [`TrainingRecipe.stages`](../saddlellm/TrainingRecipe.py#L278)<br><sub>`stages(self) -> List[str]`</sub> | method | `TrainingRecipe` 中实现`stages`的公开操作。 | `isinstance`, `list` |
| [`TrainingRecipe.compile`](../saddlellm/TrainingRecipe.py#L287)<br><sub>`compile(self, base_dir: Optional[str]=None, save_backend_plan: bool=False) -> Dict`</sub> | method | Compile this recipe into a TrainingOrchestrator-compatible dict. | `os.getcwd`, `self.stages`, `self._resolve_sources`, `self._resolve_distributed`, `self._compile_sft`, `self._compile_preference`, `self._compile_rlhf`, `self._compile_multimodal`, `self._compile_mopd`, `self._compile_vla` |
| [`TrainingRecipe.save_compiled`](../saddlellm/TrainingRecipe.py#L362)<br><sub>`save_compiled(self, path: str, base_dir: Optional[str]=None, save_backend_plan: bool=True) -> str`</sub> | method | `TrainingRecipe` 中保存`save_compiled`的公开操作。 | `self.compile`, `os.makedirs`, `os.path.dirname`, `open`, `endswith`, `path.lower`, `yaml.safe_dump`, `json.dump` |
| [`TrainingRecipe._resolve_sources`](../saddlellm/TrainingRecipe.py#L373)<br><sub>`_resolve_sources(self, base_dir: str) -> (List[Dict], Optional[List[float]])`</sub> | method | `TrainingRecipe` 中解析`resolve_sources`的内部辅助逻辑。 | `list`, `self._resolve_path`, `os.path.exists`, `DatasetManifest.load`, `manifest.to_saddle_sources`, `manifest.source_weights` |
| [`TrainingRecipe._resolve_distributed`](../saddlellm/TrainingRecipe.py#L388)<br><sub>`_resolve_distributed(self, stages: Sequence[str], save_backend_plan: bool) -> Dict`</sub> | method | `TrainingRecipe` 中解析分布式运行时的内部辅助逻辑。 | `FactoryBackendPlanner.recommend_parallelism`, `distributed.update`, `FactoryBackendPlanner.to_training_orchestrator_distributed`, `os.path.join`, `FactoryBackendPlanner.save_plan` |
| [`TrainingRecipe._compile_sft`](../saddlellm/TrainingRecipe.py#L423)<br><sub>`_compile_sft(self) -> Dict`</sub> | method | `TrainingRecipe` 中编译`compile_sft`的内部辅助逻辑。 | `self._stage_precedes`, `self._first_data_path` |
| [`TrainingRecipe._compile_preference`](../saddlellm/TrainingRecipe.py#L448)<br><sub>`_compile_preference(self) -> Dict`</sub> | method | `TrainingRecipe` 中编译`compile_preference`的内部辅助逻辑。 | `self._first_data_path`, `max` |
| [`TrainingRecipe._compile_rlhf`](../saddlellm/TrainingRecipe.py#L472)<br><sub>`_compile_rlhf(self) -> Dict`</sub> | method | `TrainingRecipe` 中编译`compile_rlhf`的内部辅助逻辑。 | `self._first_data_path`, `max` |
| [`TrainingRecipe._compile_multimodal`](../saddlellm/TrainingRecipe.py#L493)<br><sub>`_compile_multimodal(self) -> Dict`</sub> | method | `TrainingRecipe` 中编译`compile_multimodal`的内部辅助逻辑。 | `self._first_data_path` |
| [`TrainingRecipe._compile_mopd`](../saddlellm/TrainingRecipe.py#L518)<br><sub>`_compile_mopd(self) -> Dict`</sub> | method | `TrainingRecipe` 中编译`compile_mopd`的内部辅助逻辑。 | `self._first_data_path` |
| [`TrainingRecipe._compile_vla`](../saddlellm/TrainingRecipe.py#L536)<br><sub>`_compile_vla(self) -> Dict`</sub> | method | `TrainingRecipe` 中编译`compile_vla`的内部辅助逻辑。 | `self._first_data_path` |
| [`TrainingRecipe._first_data_path`](../saddlellm/TrainingRecipe.py#L578)<br><sub>`_first_data_path(self) -> str`</sub> | method | `TrainingRecipe` 中实现数据、路径的内部辅助逻辑。 | `get`, `self._resolve_sources`, `os.getcwd` |
| [`TrainingRecipe._stage_precedes`](../saddlellm/TrainingRecipe.py#L589)<br><sub>`_stage_precedes(self, before: str, after: str) -> bool`</sub> | method | `TrainingRecipe` 中实现训练阶段的内部辅助逻辑。 | `self.stages`, `stages.index` |
| [`TrainingRecipe._resolve_path`](../saddlellm/TrainingRecipe.py#L594)<br><sub>`_resolve_path(path: str, base_dir: str) -> str`</sub> | method | `TrainingRecipe` 中解析路径的内部辅助逻辑。 | `os.path.isabs`, `os.path.abspath`, `os.path.join` |

## `saddlellm/TrainingStrategyAdvisor.py`

共 16 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TrainingRoute.to_dict`](../saddlellm/TrainingStrategyAdvisor.py#L66)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `TrainingRoute` 转为可序列化字典。 | `asdict` |
| [`FeasibilityReport.to_dict`](../saddlellm/TrainingStrategyAdvisor.py#L83)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `FeasibilityReport` 转为可序列化字典。 | `asdict`, `route.to_dict` |
| [`FeasibilityReport.to_markdown`](../saddlellm/TrainingStrategyAdvisor.py#L88)<br><sub>`to_markdown(self) -> str`</sub> | method | `FeasibilityReport` 中实现`to_markdown`的公开操作。 | `lines.append`, `lines.extend`, `enumerate`, `join` |
| [`TrainingStrategyAdvisor.analyze`](../saddlellm/TrainingStrategyAdvisor.py#L135)<br><sub>`analyze(target_family: str, base_model: Optional[str]=None, domain: Optional[str]=None, budget: str='low', prefer_scratch: bool=False, teacher_models: Optional[Sequence[str]]=None) -> FeasibilityReport`</sub> | method | `TrainingStrategyAdvisor` 中分析`analyze`的公开操作。 | `BUDGETS.get`, `lower`, `TrainingStrategyAdvisor._resolve_architecture`, `list`, `TrainingStrategyAdvisor.DEFAULT_STUDENTS.get`, `bool`, `TrainingStrategyAdvisor._scratch_route`, `TrainingStrategyAdvisor._continue_route`, `TrainingStrategyAdvisor._sft_route`, `TrainingStrategyAdvisor._preference_route` |
| [`TrainingStrategyAdvisor.low_cost_recipe`](../saddlellm/TrainingStrategyAdvisor.py#L189)<br><sub>`low_cost_recipe(target_family: str, base_model: Optional[str]=None, domain: Optional[str]=None, teacher_models: Optional[Sequence[str]]=None) -> List[str]`</sub> | method | `TrainingStrategyAdvisor` 中实现训练配方的公开操作。 | `ArchitectureRegistry.normalize_name`, `TrainingStrategyAdvisor.DEFAULT_STUDENTS.get`, `join`, `TrainingStrategyAdvisor._default_teachers`, `TrainingStrategyAdvisor.DOMAIN_EVALS.get`, `lower` |
| [`TrainingStrategyAdvisor._resolve_architecture`](../saddlellm/TrainingStrategyAdvisor.py#L211)<br><sub>`_resolve_architecture(target_family: str, base_model: Optional[str]) -> ArchitectureSupport`</sub> | method | `TrainingStrategyAdvisor` 中解析`resolve_architecture`的内部辅助逻辑。 | `ArchitectureRegistry.get`, `ArchitectureRegistry.detect_from_model_name` |
| [`TrainingStrategyAdvisor._scratch_route`](../saddlellm/TrainingStrategyAdvisor.py#L220)<br><sub>`_scratch_route(support: ArchitectureSupport, budget: BudgetProfile, can_scratch: bool) -> TrainingRoute`</sub> | method | `TrainingStrategyAdvisor` 中实现路线的内部辅助逻辑。 | `TrainingRoute` |
| [`TrainingStrategyAdvisor._continue_route`](../saddlellm/TrainingStrategyAdvisor.py#L241)<br><sub>`_continue_route(support: ArchitectureSupport, base_model: Optional[str]) -> TrainingRoute`</sub> | method | `TrainingStrategyAdvisor` 中实现路线的内部辅助逻辑。 | `bool`, `TrainingRoute` |
| [`TrainingStrategyAdvisor._sft_route`](../saddlellm/TrainingStrategyAdvisor.py#L253)<br><sub>`_sft_route(support: ArchitectureSupport, base_model: Optional[str]) -> TrainingRoute`</sub> | method | `TrainingStrategyAdvisor` 中实现路线的内部辅助逻辑。 | `bool`, `TrainingRoute` |
| [`TrainingStrategyAdvisor._preference_route`](../saddlellm/TrainingStrategyAdvisor.py#L265)<br><sub>`_preference_route(support: ArchitectureSupport, base_model: Optional[str]) -> TrainingRoute`</sub> | method | `TrainingStrategyAdvisor` 中实现路线的内部辅助逻辑。 | `bool`, `TrainingRoute` |
| [`TrainingStrategyAdvisor._rl_route`](../saddlellm/TrainingStrategyAdvisor.py#L277)<br><sub>`_rl_route(support: ArchitectureSupport, base_model: Optional[str]) -> TrainingRoute`</sub> | method | `TrainingStrategyAdvisor` 中实现路线的内部辅助逻辑。 | `bool`, `TrainingRoute` |
| [`TrainingStrategyAdvisor._distill_route`](../saddlellm/TrainingStrategyAdvisor.py#L289)<br><sub>`_distill_route(support: ArchitectureSupport, base_model: Optional[str], teacher_models: Sequence[str]) -> TrainingRoute`</sub> | method | `TrainingStrategyAdvisor` 中实现路线的内部辅助逻辑。 | `bool`, `join`, `TrainingRoute` |
| [`TrainingStrategyAdvisor._blockers`](../saddlellm/TrainingStrategyAdvisor.py#L306)<br><sub>`_blockers(support: ArchitectureSupport, budget: BudgetProfile, prefer_scratch: bool) -> List[str]`</sub> | method | `TrainingStrategyAdvisor` 中实现`blockers`的内部辅助逻辑。 | `blockers.append` |
| [`TrainingStrategyAdvisor._recommended_path`](../saddlellm/TrainingStrategyAdvisor.py#L323)<br><sub>`_recommended_path(support: ArchitectureSupport, budget: BudgetProfile, prefer_scratch: bool) -> str`</sub> | method | `TrainingStrategyAdvisor` 中实现路径的内部辅助逻辑。 | — |
| [`TrainingStrategyAdvisor._notes`](../saddlellm/TrainingStrategyAdvisor.py#L335)<br><sub>`_notes(support: ArchitectureSupport, budget: BudgetProfile, domain: Optional[str]) -> List[str]`</sub> | method | `TrainingStrategyAdvisor` 中实现`notes`的内部辅助逻辑。 | `notes.append` |
| [`TrainingStrategyAdvisor._default_teachers`](../saddlellm/TrainingStrategyAdvisor.py#L349)<br><sub>`_default_teachers(family: str) -> List[str]`</sub> | method | `TrainingStrategyAdvisor` 中实现`default_teachers`的内部辅助逻辑。 | — |

## `saddlellm/UniversalDistiller.py`

共 28 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`TeacherInterface.__init__`](../saddlellm/UniversalDistiller.py#L46)<br><sub>`__init__(self, model_type: str='local', model_name: str='', api_key: Optional[str]=None, api_base: Optional[str]=None, local_model=None, local_tokenizer=None, endpoint_url: Optional[str]=None)`</sub> | method | 初始化 `TeacherInterface` 实例及其运行依赖。 | `os.environ.get` |
| [`TeacherInterface.generate`](../saddlellm/UniversalDistiller.py#L64)<br><sub>`generate(self, prompt: str, temperature: float=0.7, max_tokens: int=2048) -> str`</sub> | method | 统一生成接口。 | `self._generate_openai_compatible`, `self._generate_anthropic`, `self._generate_local`, `self._generate_endpoint`, `ValueError` |
| [`TeacherInterface.batch_generate`](../saddlellm/UniversalDistiller.py#L77)<br><sub>`batch_generate(self, prompts: List[str], temperature: float=0.7, max_tokens: int=2048, concurrency: int=5) -> List[str]`</sub> | method | 批量生成。 | `range`, `len`, `self.generate`, `results.extend`, `time.sleep` |
| [`TeacherInterface._generate_openai_compatible`](../saddlellm/UniversalDistiller.py#L89)<br><sub>`_generate_openai_compatible(self, prompt: str, temperature: float, max_tokens: int) -> str`</sub> | method | `TeacherInterface` 中生成`generate_openai_compatible`的内部辅助逻辑。 | `OpenAI`, `client.chat.completions.create`, `logger.warning`, `self._generate_local` |
| [`TeacherInterface._generate_anthropic`](../saddlellm/UniversalDistiller.py#L106)<br><sub>`_generate_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str`</sub> | method | `TeacherInterface` 中生成`generate_anthropic`的内部辅助逻辑。 | `Anthropic`, `client.messages.create`, `logger.warning` |
| [`TeacherInterface._generate_local`](../saddlellm/UniversalDistiller.py#L121)<br><sub>`_generate_local(self, prompt: str, temperature: float, max_tokens: int) -> str`</sub> | method | `TeacherInterface` 中生成`generate_local`的内部辅助逻辑。 | `self.local_model.eval`, `to`, `self.local_tokenizer`, `torch.no_grad`, `self.local_model.generate`, `self.local_tokenizer.decode`, `len`, `strip` |
| [`TeacherInterface._generate_endpoint`](../saddlellm/UniversalDistiller.py#L136)<br><sub>`_generate_endpoint(self, prompt: str, temperature: float, max_tokens: int) -> str`</sub> | method | `TeacherInterface` 中生成`generate_endpoint`的内部辅助逻辑。 | `requests.post`, `response.json` |
| [`TeacherInterface.from_openai`](../saddlellm/UniversalDistiller.py#L146)<br><sub>`from_openai(cls, model_name: str='gpt-4o', api_key: str=None)`</sub> | method | `TeacherInterface` 中实现`from_openai`的公开操作。 | `cls` |
| [`TeacherInterface.from_anthropic`](../saddlellm/UniversalDistiller.py#L150)<br><sub>`from_anthropic(cls, model_name: str='claude-sonnet-4-6', api_key: str=None)`</sub> | method | `TeacherInterface` 中实现`from_anthropic`的公开操作。 | `cls` |
| [`TeacherInterface.from_deepseek`](../saddlellm/UniversalDistiller.py#L154)<br><sub>`from_deepseek(cls, model_name: str='deepseek-chat', api_key: str=None)`</sub> | method | `TeacherInterface` 中实现`from_deepseek`的公开操作。 | `cls` |
| [`TeacherInterface.from_local`](../saddlellm/UniversalDistiller.py#L159)<br><sub>`from_local(cls, model, tokenizer)`</sub> | method | `TeacherInterface` 中实现`from_local`的公开操作。 | `cls` |
| [`DataDistiller.__init__`](../saddlellm/UniversalDistiller.py#L207)<br><sub>`__init__(self, teacher: TeacherInterface, student: Tuple[torch.nn.Module, any], config: DistillConfig=None)`</sub> | method | 初始化 `DataDistiller` 实例及其运行依赖。 | `DistillConfig`, `next`, `self.student_model.parameters` |
| [`DataDistiller.distill`](../saddlellm/UniversalDistiller.py#L221)<br><sub>`distill(self, prompts: List[str], eval_prompts: Optional[List[str]]=None, resume_from_round: int=0) -> Dict`</sub> | method | 执行蒸馏。 | `logger.info`, `len`, `range`, `self._generate_data`, `self._filter_by_quality`, `self._generated_data.extend`, `self._train_student`, `self._round_metrics.append`, `self._evaluate`, `self.student_model.save_pretrained` |
| [`DataDistiller._generate_data`](../saddlellm/UniversalDistiller.py#L283)<br><sub>`_generate_data(self, prompts: List[str], round_idx: int) -> List[Dict]`</sub> | method | 教师模型批量生成回复。 | `range`, `len`, `self.teacher.batch_generate`, `zip`, `data.append`, `logger.info`, `min` |
| [`DataDistiller._filter_by_quality`](../saddlellm/UniversalDistiller.py#L308)<br><sub>`_filter_by_quality(self, data: List[Dict]) -> List[Dict]`</sub> | method | 质量筛选: 过滤太短/太长/空泛的回复。 | `d.get`, `len`, `any`, `min`, `max`, `filtered.append` |
| [`DataDistiller._train_student`](../saddlellm/UniversalDistiller.py#L329)<br><sub>`_train_student(self, data: List[Dict], round_idx: int) -> Dict`</sub> | method | 用生成的数据训练学生模型 (SFT)。 | `train_texts.append`, `Dataset.from_list`, `dataset.map`, `LoraConfig`, `get_peft_model`, `TrainingArguments`, `Trainer`, `DataCollatorForSeq2Seq`, `trainer.train`, `len` |
| [`DataDistiller._train_student.tokenize`](../saddlellm/UniversalDistiller.py#L343)<br><sub>`tokenize(examples)`</sub> | nested function | `DataDistiller` 中分词`tokenize`的局部回调/辅助逻辑。 | `self.student_tokenizer` |
| [`DataDistiller._evaluate`](../saddlellm/UniversalDistiller.py#L385)<br><sub>`_evaluate(self, eval_prompts: List[str]) -> Dict`</sub> | method | 评估蒸馏效果。 | `self.student_model.eval`, `RejectionSampling`, `rs.generate_candidates`, `rs.select_best`, `scores.append`, `best.get`, `sum`, `len` |
| [`MultiTeacherDistiller.__init__`](../saddlellm/UniversalDistiller.py#L422)<br><sub>`__init__(self, teachers: List[TeacherInterface], student: Tuple, weights: Optional[List[float]]=None, config: DistillConfig=None)`</sub> | method | 初始化 `MultiTeacherDistiller` 实例及其运行依赖。 | `len`, `DistillConfig`, `sum` |
| [`MultiTeacherDistiller.distill`](../saddlellm/UniversalDistiller.py#L438)<br><sub>`distill(self, prompts: List[str]) -> Dict`</sub> | method | 多教师蒸馏: 1. 每个教师对同一个 prompt 生成回复 2. 用 AI 裁判选出最好的回复 3. 用最好的回复训练学生 | `logger.info`, `len`, `enumerate`, `zip`, `max`, `int`, `random.sample`, `teacher.batch_generate`, `all_data.append`, `DataDistiller` |
| [`CapabilityDistiller.__init__`](../saddlellm/UniversalDistiller.py#L555)<br><sub>`__init__(self, teacher: TeacherInterface, student_model, student_tokenizer, config: DistillConfig=None)`</sub> | method | 初始化 `CapabilityDistiller` 实例及其运行依赖。 | `DistillConfig` |
| [`CapabilityDistiller.distill`](../saddlellm/UniversalDistiller.py#L563)<br><sub>`distill(self, capability: str, num_examples: int=500, extra_seeds: Optional[List[str]]=None) -> Dict`</sub> | method | 定向蒸馏某个能力。 | `KeyError`, `list`, `self.CAPABILITY_TEMPLATES.keys`, `logger.info`, `self._generate_seeds`, `random.choice`, `template.format`, `self.teacher.generate`, `len`, `data.append` |
| [`CapabilityDistiller.distill_multiple`](../saddlellm/UniversalDistiller.py#L609)<br><sub>`distill_multiple(self, capabilities: List[str], examples_per: int=300) -> Dict`</sub> | method | 批量蒸馏多个能力。 | `self.distill` |
| [`CapabilityDistiller._generate_seeds`](../saddlellm/UniversalDistiller.py#L616)<br><sub>`_generate_seeds(self, capability: str, count: int, extra: Optional[List[str]]=None)`</sub> | method | 自动生成种子问题。 | `seeds.extend`, `seed_bank.get`, `len`, `seeds.append`, `random.choice` |
| [`AutoDistiller.__init__`](../saddlellm/UniversalDistiller.py#L662)<br><sub>`__init__(self, teacher: TeacherInterface, student_model, student_tokenizer, config: DistillConfig=None)`</sub> | method | 初始化 `AutoDistiller` 实例及其运行依赖。 | `DistillConfig` |
| [`AutoDistiller.auto_distill`](../saddlellm/UniversalDistiller.py#L670)<br><sub>`auto_distill(self, capabilities: List[str]=None, total_examples: int=1000, eval_prompts: List[str]=None, target_score: float=0.7, max_rounds: int=3) -> Dict`</sub> | method | 全自动多轮蒸馏: Round 1: 定向蒸馏核心能力 Round 2: 评估 → 补弱项 Round 3: 最终 polish | `len`, `logger.info`, `CapabilityDistiller`, `cap_distiller.distill`, `self._history.append`, `DataDistiller`, `distiller._evaluate`, `scores.get`, `self._generate_more_prompts`, `distiller.distill` |
| [`AutoDistiller._generate_more_prompts`](../saddlellm/UniversalDistiller.py#L728)<br><sub>`_generate_more_prompts(self, eval_prompts: List[str], count: int) -> List[str]`</sub> | method | 基于 eval 生成更多相关 prompts。 | `EvolInstruct`, `evolver.evolve_batch` |
| [`AutoDistiller._generate_final_prompts`](../saddlellm/UniversalDistiller.py#L734)<br><sub>`_generate_final_prompts(self, count: int) -> List[str]`</sub> | method | `AutoDistiller` 中生成`generate_final_prompts`的内部辅助逻辑。 | `random.choice`, `range` |

## `saddlellm/UnslothFineTuner.py`

共 9 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`UnslothFineTuner.__init__`](../saddlellm/UnslothFineTuner.py#L11)<br><sub>`__init__(self, model_path: str, max_seq_length: int=1024, load_in_4bit: bool=False, use_gradient_checkpointing: bool=True, device_map: str='auto')`</sub> | method | 初始化 `UnslothFineTuner` 实例及其运行依赖。 | `FastLanguageModel.from_pretrained`, `self.model.parameters`, `self.model.gradient_checkpointing_enable` |
| [`UnslothFineTuner.fit`](../saddlellm/UnslothFineTuner.py#L36)<br><sub>`fit(self, train_dataset: Dataset, eval_dataset: Optional[Dataset]=None, text_column: str='text', label_column: Optional[str]=None, training_args: Optional[TrainingArguments]=None) -> None`</sub> | method | `UnslothFineTuner` 中拟合`fit`的公开操作。 | `train_dataset.map`, `eval_dataset.map`, `TrainingArguments`, `torch.cuda.is_available`, `torch.cuda.is_bf16_supported`, `DataCollatorForLanguageModeling`, `Trainer`, `self.trainer.train` |
| [`UnslothFineTuner.fit.tokenize_function`](../saddlellm/UnslothFineTuner.py#L44)<br><sub>`tokenize_function(examples: Dict) -> Dict`</sub> | nested function | `UnslothFineTuner` 中分词`tokenize_function`的局部回调/辅助逻辑。 | `self.tokenizer`, `clone` |
| [`UnslothFineTuner.predict`](../saddlellm/UnslothFineTuner.py#L118)<br><sub>`predict(self, text: str, max_new_tokens: int=100, **generate_kwargs) -> str`</sub> | method | `UnslothFineTuner` 中预测`predict`的公开操作。 | `to`, `self.tokenizer`, `default_generate_kwargs.update`, `self.model.generate`, `self.tokenizer.decode` |
| [`UnslothFineTuner.save`](../saddlellm/UnslothFineTuner.py#L133)<br><sub>`save(self, path: str) -> None`</sub> | method | `UnslothFineTuner` 中保存`save`的公开操作。 | `self.model.save_pretrained`, `self.tokenizer.save_pretrained` |
| [`generate_base_model_training_data`](../saddlellm/UnslothFineTuner.py#L138)<br><sub>`generate_base_model_training_data(size: int=100) -> Dataset`</sub> | function | 生成用于训练基座模型的示例语料 | `texts.extend`, `range`, `len`, `torch.randint`, `random.shuffle`, `texts.append`, `join`, `Dataset.from_dict` |
| [`gen_textsDataset_from_list`](../saddlellm/UnslothFineTuner.py#L174)<br><sub>`gen_textsDataset_from_list(text_list)`</sub> | function | 生成用于训练基座模型的示例语料 | `Dataset.from_dict` |
| [`test_train`](../saddlellm/UnslothFineTuner.py#L184)<br><sub>`test_train(train_csvdata_path='E:\\ml_data\\chinese-poetry-collection\\train.csv', eval_csvdata_path='E:\\ml_data\\chinese-poetry-collection\\test.csv', model_path='E:\\reactflow_test\\backend\\model\\Qwen\\Qwen2___5-0___5B-Instruct', num_train_epochs=2, learning_rate=1e-05, output_dir='./output_base_model', log_dir='./logs', save_steps=400, test_prompts=['你是谁', '床前明月光', '美国在哪里', '历史研究可以帮助我们'], test_prompts_generate=250, per_device_train_batch_size=2, per_device_eval_batch_size=2)`</sub> | function | 模块级训练`test_train`的公开操作。 | `pd.read_csv`, `Dataset.from_dict`, `tolist`, `print`, `len`, `UnslothFineTuner`, `version.parse`, `TrainingArguments`, `str`, `ft.fit` |
| [`evaluate_checkpoint`](../saddlellm/UnslothFineTuner.py#L283)<br><sub>`evaluate_checkpoint(checkpoint_dir='./output_base_model', eval_dataset=None)`</sub> | function | 模块级评估检查点的公开操作。 | `os.listdir`, `d.startswith`, `max`, `int`, `x.split`, `os.path.join`, `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, `Trainer`, `DataCollatorForLanguageModeling` |

## `saddlellm/UnslothSFTTrainer.py`

共 4 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`UnslotSFTTrainer`](../saddlellm/UnslothSFTTrainer.py#L5)<br><sub>`UnslotSFTTrainer(MODEL_PATH, DATASET_PATH, OUTPUT_PATH, max_seq_length=2048, random_state=3407)`</sub> | function | 模块级实现`UnslotSFTTrainer`的公开操作。 | `torch.backends.cuda.enable_flash_sdp`, `torch.backends.cuda.enable_mem_efficient_sdp`, `FastLanguageModel.from_pretrained`, `FastLanguageModel.get_peft_model`, `load_dataset`, `print`, `dataset.map`, `TrainingArguments`, `SaveEveryNStepsCallback`, `SFTTrainer` |
| [`UnslotSFTTrainer.SaveEveryNStepsCallback.__init__`](../saddlellm/UnslothSFTTrainer.py#L26)<br><sub>`__init__(self, save_steps=40, output_dir=OUTPUT_PATH)`</sub> | nested function | 初始化 `SaveEveryNStepsCallback` 实例及其运行依赖。 | — |
| [`UnslotSFTTrainer.SaveEveryNStepsCallback.on_step_end`](../saddlellm/UnslothSFTTrainer.py#L30)<br><sub>`on_step_end(self, args, state, control, model=None, **kwargs)`</sub> | nested function | `SaveEveryNStepsCallback` 中实现`on_step_end`的局部回调/辅助逻辑。 | `print`, `model.save_pretrained`, `tokenizer.save_pretrained`, `torch.cuda.empty_cache`, `gc.collect`, `torch.cuda.is_available`, `torch.cuda.memory_allocated`, `torch.cuda.memory_reserved` |
| [`UnslotSFTTrainer.formatting_prompts_func`](../saddlellm/UnslothSFTTrainer.py#L78)<br><sub>`formatting_prompts_func(examples)`</sub> | nested function | 模块级实现`formatting_prompts_func`的局部回调/辅助逻辑。 | `zip`, `alpaca_prompt.format`, `texts.append` |

## `saddlellm/VLA.py`

共 38 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`VLAActionSpace.to_dict`](../saddlellm/VLA.py#L30)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VLAActionSpace` 转为可序列化字典。 | `asdict` |
| [`VLAAction.to_dict`](../saddlellm/VLA.py#L42)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VLAAction` 转为可序列化字典。 | `asdict` |
| [`VLASample.to_dict`](../saddlellm/VLA.py#L56)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VLASample` 转为可序列化字典。 | `self.action.to_dict`, `image.to_dict` |
| [`VLANormalizationReport.to_dict`](../saddlellm/VLA.py#L78)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VLANormalizationReport` 转为可序列化字典。 | `asdict` |
| [`VLACollatorConfig.to_dict`](../saddlellm/VLA.py#L91)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VLACollatorConfig` 转为可序列化字典。 | `asdict` |
| [`VLAActionTokenizer.__init__`](../saddlellm/VLA.py#L98)<br><sub>`__init__(self, action_space: Optional[VLAActionSpace]=None)`</sub> | method | 初始化 `VLAActionTokenizer` 实例及其运行依赖。 | `VLAActionSpace` |
| [`VLAActionTokenizer.encode`](../saddlellm/VLA.py#L101)<br><sub>`encode(self, action: Union[VLAAction, Sequence[float], str]) -> str`</sub> | method | `VLAActionTokenizer` 中编码`encode`的公开操作。 | `isinstance`, `VLAAction`, `float`, `list`, `values.append`, `self._token_for_value`, `join` |
| [`VLAActionTokenizer.decode`](../saddlellm/VLA.py#L114)<br><sub>`decode(self, text: str) -> VLAAction`</sub> | method | `VLAActionTokenizer` 中解码`decode`的公开操作。 | `re.escape`, `int`, `re.findall`, `self._value_for_bin`, `len`, `VLAAction` |
| [`VLAActionTokenizer.action_vocab`](../saddlellm/VLA.py#L124)<br><sub>`action_vocab(self) -> List[str]`</sub> | method | `VLAActionTokenizer` 中实现动作的公开操作。 | `range` |
| [`VLAActionTokenizer.special_tokens`](../saddlellm/VLA.py#L130)<br><sub>`special_tokens(self, include_image_token: bool=True, image_token: str='<image>') -> List[str]`</sub> | method | `VLAActionTokenizer` 中实现`special_tokens`的公开操作。 | `self.action_vocab` |
| [`VLAActionTokenizer.register_with_tokenizer`](../saddlellm/VLA.py#L136)<br><sub>`register_with_tokenizer(self, tokenizer: Any, include_image_token: bool=True, image_token: str='<image>', resize_model: Optional[Any]=None) -> Dict`</sub> | method | `VLAActionTokenizer` 中注册分词器的公开操作。 | `self.special_tokens`, `hasattr`, `tokenizer.add_special_tokens`, `tokenizer.add_tokens`, `TypeError`, `resize_model.resize_token_embeddings`, `len`, `tokenizer.convert_tokens_to_ids`, `int`, `self.action_vocab` |
| [`VLAActionTokenizer._token_for_value`](../saddlellm/VLA.py#L163)<br><sub>`_token_for_value(self, value: float) -> str`</sub> | method | `VLAActionTokenizer` 中实现`token_for_value`的内部辅助逻辑。 | `min`, `max`, `float`, `int`, `round` |
| [`VLAActionTokenizer._value_for_bin`](../saddlellm/VLA.py#L171)<br><sub>`_value_for_bin(self, bin_id: int) -> float`</sub> | method | `VLAActionTokenizer` 中实现`value_for_bin`的内部辅助逻辑。 | `min`, `max`, `int` |
| [`VLADataAdapter.normalize_record`](../saddlellm/VLA.py#L183)<br><sub>`normalize_record(cls, record: Dict, image_root: Optional[str]=None, action_tokenizer: Optional[VLAActionTokenizer]=None) -> Optional[VLASample]`</sub> | method | `VLADataAdapter` 中规范化`normalize_record`的公开操作。 | `VLAActionTokenizer`, `cls._first`, `cls._extract_action`, `cls._extract_images`, `cls._extract_proprio`, `tokenizer.encode`, `isinstance`, `record.get`, `dict`, `VLASample` |
| [`VLADataAdapter.normalize_records`](../saddlellm/VLA.py#L220)<br><sub>`normalize_records(cls, records: Iterable[Dict], image_root: Optional[str]=None, action_space: Optional[VLAActionSpace]=None) -> (List[VLASample], VLANormalizationReport)`</sub> | method | `VLADataAdapter` 中规范化`normalize_records`的公开操作。 | `VLAActionTokenizer`, `VLANormalizationReport`, `cls._expand_trajectory_record`, `cls.detect_schema`, `report.detected_schemas.get`, `cls.normalize_record`, `samples.append`, `report.warnings.append` |
| [`VLADataAdapter.normalize_file`](../saddlellm/VLA.py#L246)<br><sub>`normalize_file(cls, input_path: str, output_path: str, image_root: Optional[str]=None, action_space: Optional[VLAActionSpace]=None) -> VLANormalizationReport`</sub> | method | `VLADataAdapter` 中规范化`normalize_file`的公开操作。 | `cls.load_records`, `cls.normalize_records`, `os.path.dirname`, `os.makedirs`, `open`, `f.write`, `json.dumps`, `sample.to_dict`, `json.dump`, `report.to_dict` |
| [`VLADataAdapter.load_records`](../saddlellm/VLA.py#L270)<br><sub>`load_records(path: str) -> List[Dict]`</sub> | method | `VLADataAdapter` 中加载`load_records`的公开操作。 | `lower`, `os.path.splitext`, `open`, `line.strip`, `rows.append`, `json.loads`, `json.load`, `isinstance`, `data.get`, `list` |
| [`VLADataAdapter.detect_schema`](../saddlellm/VLA.py#L295)<br><sub>`detect_schema(record: Dict) -> str`</sub> | method | `VLADataAdapter` 中检测`detect_schema`的公开操作。 | `any` |
| [`VLADataAdapter._expand_trajectory_record`](../saddlellm/VLA.py#L309)<br><sub>`_expand_trajectory_record(cls, record: Dict) -> List[Dict]`</sub> | method | `VLADataAdapter` 中记录`expand_trajectory_record`的内部辅助逻辑。 | `record.get`, `isinstance`, `record.items`, `enumerate`, `dict`, `item.update`, `item.setdefault`, `step.get`, `expanded.append` |
| [`VLADataAdapter._extract_action`](../saddlellm/VLA.py#L333)<br><sub>`_extract_action(cls, record: Dict) -> Optional[VLAAction]`</sub> | method | `VLADataAdapter` 中提取动作的内部辅助逻辑。 | `record.get`, `isinstance`, `VLAAction`, `raw.get`, `json.loads`, `raw.items`, `float` |
| [`VLADataAdapter._extract_images`](../saddlellm/VLA.py#L360)<br><sub>`_extract_images(cls, record: Dict, image_root: Optional[str]) -> List[MultimodalAsset]`</sub> | method | `VLADataAdapter` 中提取`extract_images`的内部辅助逻辑。 | `isinstance`, `record.get`, `observation.get`, `item.get`, `item.items`, `str`, `os.path.isabs`, `path.startswith`, `os.path.abspath`, `os.path.join` |
| [`VLADataAdapter._extract_proprio`](../saddlellm/VLA.py#L390)<br><sub>`_extract_proprio(record: Dict) -> Dict`</sub> | method | `VLADataAdapter` 中提取`extract_proprio`的内部辅助逻辑。 | `isinstance`, `record.get`, `observation.get`, `dict` |
| [`VLADataAdapter._first`](../saddlellm/VLA.py#L396)<br><sub>`_first(record: Dict, keys: Sequence[str]) -> str`</sub> | method | `VLADataAdapter` 中实现`first`的内部辅助逻辑。 | `record.get`, `str`, `isinstance`, `observation.get` |
| [`VLADataset.__init__`](../saddlellm/VLA.py#L412)<br><sub>`__init__(self, samples: Sequence[Dict])`</sub> | method | 初始化 `VLADataset` 实例及其运行依赖。 | `list` |
| [`VLADataset.__len__`](../saddlellm/VLA.py#L415)<br><sub>`__len__(self) -> int`</sub> | method | `VLADataset` 中实现`len__`的内部辅助逻辑。 | `len` |
| [`VLADataset.__getitem__`](../saddlellm/VLA.py#L418)<br><sub>`__getitem__(self, idx: int) -> Dict`</sub> | method | `VLADataset` 中实现`getitem__`的内部辅助逻辑。 | — |
| [`VLADataset.from_file`](../saddlellm/VLA.py#L422)<br><sub>`from_file(cls, path: str) -> 'VLADataset'`</sub> | method | `VLADataset` 中实现`from_file`的公开操作。 | `open`, `line.strip`, `rows.append`, `json.loads`, `cls` |
| [`VLADataCollator.__init__`](../saddlellm/VLA.py#L440)<br><sub>`__init__(self, tokenizer: Any, config: Optional[VLACollatorConfig]=None, action_tokenizer: Optional[VLAActionTokenizer]=None)`</sub> | method | 初始化 `VLADataCollator` 实例及其运行依赖。 | `VLACollatorConfig`, `VLAActionTokenizer` |
| [`VLADataCollator.__call__`](../saddlellm/VLA.py#L450)<br><sub>`__call__(self, features: Sequence[Dict]) -> Dict`</sub> | method | `VLADataCollator` 中实现`call__`的内部辅助逻辑。 | `self._encode_feature`, `max`, `len`, `self._pad_token_id`, `list`, `min`, `input_ids.append`, `attention_mask.append`, `labels.append`, `torch.tensor` |
| [`VLADataCollator._encode_feature`](../saddlellm/VLA.py#L476)<br><sub>`_encode_feature(self, feature: Dict) -> Tuple[List[int], int]`</sub> | method | `VLADataCollator` 中编码`encode_feature`的内部辅助逻辑。 | `feature.get`, `self._messages_from_feature`, `self._format_messages`, `self._tokenize`, `len`, `max` |
| [`VLADataCollator._format_messages`](../saddlellm/VLA.py#L489)<br><sub>`_format_messages(self, messages: Sequence[Dict[str, str]]) -> Tuple[str, str]`</sub> | method | `VLADataCollator` 中格式化`format_messages`的内部辅助逻辑。 | `range`, `len`, `get`, `join`, `self._format_line`, `strip`, `str` |
| [`VLADataCollator._format_line`](../saddlellm/VLA.py#L506)<br><sub>`_format_line(message: Dict[str, str]) -> str`</sub> | method | `VLADataCollator` 中格式化`format_line`的内部辅助逻辑。 | `strip`, `str`, `message.get` |
| [`VLADataCollator._messages_from_feature`](../saddlellm/VLA.py#L511)<br><sub>`_messages_from_feature(self, feature: Dict) -> List[Dict[str, str]]`</sub> | method | `VLADataCollator` 中实现`messages_from_feature`的内部辅助逻辑。 | `strip`, `str`, `feature.get`, `isinstance`, `action.get`, `self.action_tokenizer.encode` |
| [`VLADataCollator._tokenize`](../saddlellm/VLA.py#L525)<br><sub>`_tokenize(self, text: str) -> List[int]`</sub> | method | `VLADataCollator` 中分词`tokenize`的内部辅助逻辑。 | `self.tokenizer`, `isinstance`, `int` |
| [`VLADataCollator._pad_token_id`](../saddlellm/VLA.py#L532)<br><sub>`_pad_token_id(self) -> int`</sub> | method | `VLADataCollator` 中实现`pad_token_id`的内部辅助逻辑。 | `getattr`, `int` |
| [`VLATrainingPlanner.create_plan`](../saddlellm/VLA.py#L544)<br><sub>`create_plan(data_path: Optional[str]=None, image_root: Optional[str]=None, action_space: Optional[VLAActionSpace]=None, output_dir: str='./vla_plan') -> Dict`</sub> | method | `VLATrainingPlanner` 中创建计划的公开操作。 | `VLAActionSpace`, `VLAActionTokenizer`, `to_dict`, `VLACollatorConfig`, `action_space.to_dict`, `len`, `tokenizer.action_vocab`, `tokenizer.special_tokens` |
| [`normalize_vla_file`](../saddlellm/VLA.py#L573)<br><sub>`normalize_vla_file(input_path: str, output_path: str, image_root: Optional[str]=None, action_space: Optional[VLAActionSpace]=None) -> Dict`</sub> | function | 模块级规范化`normalize_vla_file`的公开操作。 | `to_dict`, `VLADataAdapter.normalize_file` |
| [`register_vla_action_tokens`](../saddlellm/VLA.py#L587)<br><sub>`register_vla_action_tokens(tokenizer: Any, action_space: Optional[VLAActionSpace]=None, include_image_token: bool=True, image_token: str='<image>', resize_model: Optional[Any]=None) -> Dict`</sub> | function | 模块级注册动作的公开操作。 | `register_with_tokenizer`, `VLAActionTokenizer` |

## `saddlellm/VLADataInspector.py`

共 12 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`VLADataInspection.usable_ratio`](../saddlellm/VLADataInspector.py#L33)<br><sub>`usable_ratio(self) -> float`</sub> | method | `VLADataInspection` 中实现`usable_ratio`的公开操作。 | — |
| [`VLADataInspection.ready`](../saddlellm/VLADataInspector.py#L37)<br><sub>`ready(self) -> bool`</sub> | method | `VLADataInspection` 中实现`ready`的公开操作。 | — |
| [`VLADataInspection.severity`](../saddlellm/VLADataInspector.py#L41)<br><sub>`severity(self) -> str`</sub> | method | `VLADataInspection` 中实现`severity`的公开操作。 | — |
| [`VLADataInspection.to_dict`](../saddlellm/VLADataInspector.py#L48)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VLADataInspection` 转为可序列化字典。 | `asdict` |
| [`VLADataInspector.inspect_file`](../saddlellm/VLADataInspector.py#L60)<br><sub>`inspect_file(cls, path: str, image_root: Optional[str]=None, action_space: Optional[VLAActionSpace]=None, max_records: int=256) -> VLADataInspection`</sub> | method | `VLADataInspector` 中检查`inspect_file`的公开操作。 | `VLADataInspection`, `VLAActionSpace`, `VLAActionTokenizer`, `report.errors.append`, `report.recommendations.append`, `os.path.exists`, `os.path.isfile`, `report.warnings.append`, `os.path.dirname`, `os.path.abspath` |
| [`VLADataInspector._sample_records`](../saddlellm/VLADataInspector.py#L148)<br><sub>`_sample_records(records: Iterable[Dict], max_records: int) -> Iterable[Dict]`</sub> | method | `VLADataInspector` 中采样`sample_records`的内部辅助逻辑。 | `enumerate`, `isinstance` |
| [`VLADataInspector._inspect_action`](../saddlellm/VLADataInspector.py#L156)<br><sub>`_inspect_action(action: VLAAction, action_space: VLAActionSpace) -> Tuple[bool, int, bool]`</sub> | method | `VLADataInspector` 中检查动作的内部辅助逻辑。 | `list`, `values.append`, `float`, `expected_lengths.add`, `len`, `any` |
| [`VLADataInspector._missing_local_images`](../saddlellm/VLADataInspector.py#L170)<br><sub>`_missing_local_images(images) -> List[str]`</sub> | method | `VLADataInspector` 中实现`missing_local_images`的内部辅助逻辑。 | `getattr`, `path.startswith`, `os.path.exists`, `missing.append` |
| [`VLADataInspector._record_episode_step`](../saddlellm/VLADataInspector.py#L181)<br><sub>`_record_episode_step(record: Dict, episodes: Dict[str, List[int]]) -> None`</sub> | method | `VLADataInspector` 中记录`record_episode_step`的内部辅助逻辑。 | `record.get`, `append`, `episodes.setdefault`, `str`, `int` |
| [`VLADataInspector._count_step_warnings`](../saddlellm/VLADataInspector.py#L192)<br><sub>`_count_step_warnings(episodes: Dict[str, List[int]]) -> int`</sub> | method | `VLADataInspector` 中实现`count_step_warnings`的内部辅助逻辑。 | `episodes.values`, `len`, `sorted`, `set`, `list`, `range` |
| [`VLADataInspector._finalize`](../saddlellm/VLADataInspector.py#L204)<br><sub>`_finalize(report: VLADataInspection) -> None`</sub> | method | `VLADataInspector` 中实现`finalize`的内部辅助逻辑。 | `report.errors.append`, `report.recommendations.append`, `report.warnings.append` |
| [`inspect_vla_data`](../saddlellm/VLADataInspector.py#L228)<br><sub>`inspect_vla_data(path: str, image_root: Optional[str]=None, action_space: Optional[VLAActionSpace]=None, max_records: int=256) -> Dict`</sub> | function | 模块级检查数据的公开操作。 | `to_dict`, `VLADataInspector.inspect_file` |

## `saddlellm/VLATrainer.py`

共 14 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`VLASFTConfig.to_dict`](../saddlellm/VLATrainer.py#L63)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VLASFTConfig` 转为可序列化字典。 | `asdict`, `self.action_space.to_dict` |
| [`train_vla_sft`](../saddlellm/VLATrainer.py#L69)<br><sub>`train_vla_sft(config: VLASFTConfig) -> Dict`</sub> | function | 模块级训练`train_vla_sft`的公开操作。 | `os.makedirs`, `_ensure_normalized_dataset`, `VLADataset.from_file`, `len`, `ValueError`, `_split_dataset`, `_device`, `AutoTokenizer.from_pretrained`, `_dtype_for_device`, `_maybe_quantization_config` |
| [`_ensure_normalized_dataset`](../saddlellm/VLATrainer.py#L152)<br><sub>`_ensure_normalized_dataset(config: VLASFTConfig) -> str`</sub> | function | 模块级确保数据集的内部辅助逻辑。 | `_looks_normalized`, `os.path.join`, `VLADataAdapter.normalize_file` |
| [`_looks_normalized`](../saddlellm/VLATrainer.py#L165)<br><sub>`_looks_normalized(path: str) -> bool`</sub> | function | 模块级实现`looks_normalized`的内部辅助逻辑。 | `os.path.exists`, `open`, `endswith`, `path.lower`, `line.strip`, `json.loads`, `_record_is_normalized`, `json.load`, `isinstance` |
| [`_record_is_normalized`](../saddlellm/VLATrainer.py#L187)<br><sub>`_record_is_normalized(record: Dict) -> bool`</sub> | function | 模块级记录`record_is_normalized`的内部辅助逻辑。 | `bool`, `record.get` |
| [`_split_dataset`](../saddlellm/VLATrainer.py#L191)<br><sub>`_split_dataset(dataset: VLADataset, validation_split: float, seed: int)`</sub> | function | 模块级切分数据集的内部辅助逻辑。 | `len`, `list`, `range`, `shuffle`, `random.Random`, `max`, `int`, `round`, `Subset` |
| [`_model_input_collator`](../saddlellm/VLATrainer.py#L202)<br><sub>`_model_input_collator(collator: VLADataCollator)`</sub> | function | 模块级实现模型、输入的内部辅助逻辑。 | — |
| [`_model_input_collator.wrapped`](../saddlellm/VLATrainer.py#L203)<br><sub>`wrapped(features)`</sub> | nested function | 模块级实现`wrapped`的局部回调/辅助逻辑。 | `collator` |
| [`_device`](../saddlellm/VLATrainer.py#L214)<br><sub>`_device() -> str`</sub> | function | 模块级实现设备的内部辅助逻辑。 | `torch.cuda.is_available`, `hasattr`, `torch.backends.mps.is_available` |
| [`_supports_bf16`](../saddlellm/VLATrainer.py#L222)<br><sub>`_supports_bf16() -> bool`</sub> | function | 模块级实现`supports_bf16`的内部辅助逻辑。 | `torch.cuda.is_available`, `torch.cuda.is_bf16_supported` |
| [`_dtype_for_device`](../saddlellm/VLATrainer.py#L226)<br><sub>`_dtype_for_device(device: str)`</sub> | function | 模块级实现设备的内部辅助逻辑。 | `_supports_bf16` |
| [`_maybe_quantization_config`](../saddlellm/VLATrainer.py#L232)<br><sub>`_maybe_quantization_config(config: VLASFTConfig, device: str)`</sub> | function | 模块级实现配置的内部辅助逻辑。 | `BitsAndBytesConfig`, `_supports_bf16`, `logger.warning` |
| [`_maybe_wrap_lora`](../saddlellm/VLATrainer.py#L249)<br><sub>`_maybe_wrap_lora(model, config: VLASFTConfig)`</sub> | function | 模块级实现`maybe_wrap_lora`的内部辅助逻辑。 | `LoraConfig`, `get_peft_model`, `logger.warning` |
| [`_make_training_args`](../saddlellm/VLATrainer.py#L267)<br><sub>`_make_training_args(config: VLASFTConfig, device: str) -> TrainingArguments`</sub> | function | 模块级实现训练的内部辅助逻辑。 | `_supports_bf16`, `set`, `inspect.signature`, `kwargs.items`, `TrainingArguments` |

## `saddlellm/VideoLatentFlowModel.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`VideoLatentFlowConfig.__post_init__`](../saddlellm/VideoLatentFlowModel.py#L29)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `VideoLatentFlowConfig` 创建后校验并规范化字段。 | `int`, `getattr`, `ValueError` |
| [`VideoLatentFlowConfig.temporal_tokens`](../saddlellm/VideoLatentFlowModel.py#L54)<br><sub>`temporal_tokens(self) -> int`</sub> | method | `VideoLatentFlowConfig` 中实现`temporal_tokens`的公开操作。 | — |
| [`VideoLatentFlowConfig.spatial_tokens`](../saddlellm/VideoLatentFlowModel.py#L58)<br><sub>`spatial_tokens(self) -> int`</sub> | method | `VideoLatentFlowConfig` 中实现`spatial_tokens`的公开操作。 | — |
| [`VideoLatentFlowConfig.patch_dim`](../saddlellm/VideoLatentFlowModel.py#L64)<br><sub>`patch_dim(self) -> int`</sub> | method | `VideoLatentFlowConfig` 中实现`patch_dim`的公开操作。 | — |
| [`VideoLatentFlowConfig.to_dict`](../saddlellm/VideoLatentFlowModel.py#L72)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `VideoLatentFlowConfig` 转为可序列化字典。 | `asdict` |
| [`ConditionalVideoLatentFlowTransformer.__init__`](../saddlellm/VideoLatentFlowModel.py#L79)<br><sub>`__init__(self, config: VideoLatentFlowConfig) -> None`</sub> | method | 初始化 `ConditionalVideoLatentFlowTransformer` 实例及其运行依赖。 | `__init__`, `super`, `nn.Conv3d`, `nn.Parameter`, `torch.zeros`, `nn.Sequential`, `nn.LayerNorm`, `nn.Linear`, `nn.SiLU`, `nn.TransformerEncoderLayer` |
| [`ConditionalVideoLatentFlowTransformer.forward`](../saddlellm/VideoLatentFlowModel.py#L128)<br><sub>`forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `ConditionalVideoLatentFlowTransformer` 的前向计算。 | `self._validate_inputs`, `self.patch_embed`, `permute`, `embedded.flatten`, `self.temporal_position.to`, `self.spatial_position.to`, `tokens.reshape`, `to`, `self.time_projection`, `_timestep_embedding` |
| [`ConditionalVideoLatentFlowTransformer.compute_flow_loss`](../saddlellm/VideoLatentFlowModel.py#L150)<br><sub>`compute_flow_loss(self, target_latents: torch.Tensor, condition: torch.Tensor, *, noise: Optional[torch.Tensor]=None, timesteps: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]`</sub> | method | `ConditionalVideoLatentFlowTransformer` 中计算流程的公开操作。 | `torch.randn_like`, `ValueError`, `torch.rand`, `timesteps.reshape`, `self`, `F.mse_loss`, `prediction.float`, `target_velocity.float`, `loss.detach`, `torch.sqrt` |
| [`ConditionalVideoLatentFlowTransformer.sample`](../saddlellm/VideoLatentFlowModel.py#L178)<br><sub>`sample(self, condition: torch.Tensor, *, num_steps: int=30, initial_noise: Optional[torch.Tensor]=None) -> torch.Tensor`</sub> | method | `ConditionalVideoLatentFlowTransformer` 中采样`sample`的公开操作。 | `ValueError`, `next`, `self.parameters`, `torch.randn`, `tuple`, `initial_noise.clone`, `self.eval`, `range`, `torch.full`, `self` |
| [`ConditionalVideoLatentFlowTransformer._unpatchify`](../saddlellm/VideoLatentFlowModel.py#L219)<br><sub>`_unpatchify(self, patches: torch.Tensor) -> torch.Tensor`</sub> | method | `ConditionalVideoLatentFlowTransformer` 中实现`unpatchify`的内部辅助逻辑。 | `patches.reshape`, `reshape`, `patches.permute` |
| [`ConditionalVideoLatentFlowTransformer._validate_inputs`](../saddlellm/VideoLatentFlowModel.py#L243)<br><sub>`_validate_inputs(self, latents: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> None`</sub> | method | `ConditionalVideoLatentFlowTransformer` 中校验`validate_inputs`的内部辅助逻辑。 | `tuple`, `ValueError` |

## `saddlellm/VideoLatentFlowTrainer.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`VideoLatentFlowTrainingConfig.__post_init__`](../saddlellm/VideoLatentFlowTrainer.py#L42)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `VideoLatentFlowTrainingConfig` 创建后校验并规范化字段。 | `int`, `getattr`, `ValueError` |
| [`CachedVideoLatentDataset.__init__`](../saddlellm/VideoLatentFlowTrainer.py#L59)<br><sub>`__init__(self, path: str) -> None`</sub> | method | 初始化 `CachedVideoLatentDataset` 实例及其运行依赖。 | `Path`, `source.is_dir`, `source.suffix.lower`, `ShardedNpzStore`, `self._sharded.array_shape`, `source.is_file`, `FileNotFoundError`, `ValueError`, `np.load`, `set` |
| [`CachedVideoLatentDataset.__len__`](../saddlellm/VideoLatentFlowTrainer.py#L99)<br><sub>`__len__(self) -> int`</sub> | method | `CachedVideoLatentDataset` 中实现`len__`的内部辅助逻辑。 | `len` |
| [`CachedVideoLatentDataset.__getitem__`](../saddlellm/VideoLatentFlowTrainer.py#L102)<br><sub>`__getitem__(self, index: int) -> Dict[str, torch.Tensor]`</sub> | method | `CachedVideoLatentDataset` 中实现`getitem__`的内部辅助逻辑。 | `self._sharded.get`, `float`, `torch.from_numpy`, `np.array` |
| [`CachedVideoLatentDataset.infer_model_config`](../saddlellm/VideoLatentFlowTrainer.py#L118)<br><sub>`infer_model_config(self, **overrides: Any) -> VideoLatentFlowConfig`</sub> | method | `CachedVideoLatentDataset` 中推断模型、配置的公开操作。 | `VideoLatentFlowConfig`, `int` |
| [`_resolve_device`](../saddlellm/VideoLatentFlowTrainer.py#L130)<br><sub>`_resolve_device(value: str) -> torch.device`</sub> | function | 模块级解析设备的内部辅助逻辑。 | `torch.device`, `torch.cuda.is_available`, `RuntimeError` |
| [`_autocast`](../saddlellm/VideoLatentFlowTrainer.py#L139)<br><sub>`_autocast(device: torch.device, precision: str)`</sub> | function | 模块级实现`autocast`的内部辅助逻辑。 | `torch.autocast` |
| [`_lr_multiplier`](../saddlellm/VideoLatentFlowTrainer.py#L146)<br><sub>`_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float`</sub> | function | 模块级实现`lr_multiplier`的内部辅助逻辑。 | `max`, `math.cos`, `min` |
| [`save_video_latent_flow_checkpoint`](../saddlellm/VideoLatentFlowTrainer.py#L153)<br><sub>`save_video_latent_flow_checkpoint(model: ConditionalVideoLatentFlowTransformer, path: str, *, training: Optional[VideoLatentFlowTrainingConfig]=None, global_step: int=0, optimizer: Optional[torch.optim.Optimizer]=None, scheduler: Optional[torch.optim.lr_scheduler.LambdaLR]=None) -> str`</sub> | function | 模块级保存流程、检查点的公开操作。 | `os.makedirs`, `torch.save`, `model.state_dict`, `os.path.join`, `model.config.to_dict`, `int`, `asdict`, `open`, `json.dump`, `file.write` |
| [`load_video_latent_flow_checkpoint`](../saddlellm/VideoLatentFlowTrainer.py#L190)<br><sub>`load_video_latent_flow_checkpoint(path: str, *, map_location: str='cpu') -> ConditionalVideoLatentFlowTransformer`</sub> | function | 模块级加载流程、检查点的公开操作。 | `Path`, `config_path.is_file`, `weights_path.is_file`, `FileNotFoundError`, `json.loads`, `config_path.read_text`, `ConditionalVideoLatentFlowTransformer`, `VideoLatentFlowConfig`, `model.load_state_dict`, `torch.load` |
| [`train_video_latent_flow`](../saddlellm/VideoLatentFlowTrainer.py#L211)<br><sub>`train_video_latent_flow(model_config: VideoLatentFlowConfig, training_config: VideoLatentFlowTrainingConfig) -> Dict[str, Any]`</sub> | function | 模块级训练流程的公开操作。 | `random.seed`, `np.random.seed`, `torch.manual_seed`, `CachedVideoLatentDataset`, `ValueError`, `_resolve_device`, `to`, `ConditionalVideoLatentFlowTransformer`, `DataLoader`, `math.ceil` |

## `saddlellm/VisionBackbones.py`

共 9 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`VisionBackboneConfig.to_dict`](../saddlellm/VisionBackbones.py#L23)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VisionBackboneConfig` 转为可序列化字典。 | `asdict` |
| [`VisionBackboneSpec.to_dict`](../saddlellm/VisionBackbones.py#L36)<br><sub>`to_dict(self) -> Dict`</sub> | method | 把 `VisionBackboneSpec` 转为可序列化字典。 | `asdict` |
| [`TinyPatchVisionBackbone.__init__`](../saddlellm/VisionBackbones.py#L43)<br><sub>`__init__(self, config: VisionBackboneConfig)`</sub> | method | 初始化 `TinyPatchVisionBackbone` 实例及其运行依赖。 | `__init__`, `super`, `nn.Conv2d`, `nn.LayerNorm` |
| [`TinyPatchVisionBackbone.forward`](../saddlellm/VisionBackbones.py#L56)<br><sub>`forward(self, pixel_values: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `TinyPatchVisionBackbone` 的前向计算。 | `self.patch_embed`, `contiguous`, `transpose`, `features.flatten`, `self.norm` |
| [`HFVisionBackbone.__init__`](../saddlellm/VisionBackbones.py#L67)<br><sub>`__init__(self, config: VisionBackboneConfig)`</sub> | method | 初始化 `HFVisionBackbone` 实例及其运行依赖。 | `__init__`, `super`, `importlib.util.find_spec`, `ImportError`, `AutoModel.from_pretrained`, `getattr`, `int`, `self.model.parameters`, `param.requires_grad_` |
| [`HFVisionBackbone.forward`](../saddlellm/VisionBackbones.py#L87)<br><sub>`forward(self, pixel_values: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `HFVisionBackbone` 的前向计算。 | `self.model`, `hasattr`, `isinstance`, `ValueError` |
| [`VisionBackboneRegistry.list_backbones`](../saddlellm/VisionBackbones.py#L104)<br><sub>`list_backbones() -> Dict[str, Dict]`</sub> | method | `VisionBackboneRegistry` 中列出`list_backbones`的公开操作。 | `importlib.util.find_spec`, `to_dict`, `VisionBackboneSpec` |
| [`VisionBackboneRegistry.build`](../saddlellm/VisionBackbones.py#L142)<br><sub>`build(config: VisionBackboneConfig) -> nn.Module`</sub> | method | `VisionBackboneRegistry` 中构建`build`的公开操作。 | `lower`, `TinyPatchVisionBackbone`, `HFVisionBackbone`, `model.parameters`, `param.requires_grad_` |
| [`list_vision_backbones`](../saddlellm/VisionBackbones.py#L154)<br><sub>`list_vision_backbones() -> Dict[str, Dict]`</sub> | function | 模块级列出`list_vision_backbones`的公开操作。 | `VisionBackboneRegistry.list_backbones` |

## `saddlellm/WorldAgent.py`

共 62 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`WorldAgentError.__init__`](../saddlellm/WorldAgent.py#L57)<br><sub>`__init__(self, message: str, status_code: int=400, code: str='invalid_request') -> None`</sub> | method | 初始化 `WorldAgentError` 实例及其运行依赖。 | `__init__`, `super`, `int`, `str` |
| [`WorldAgentSettings.__post_init__`](../saddlellm/WorldAgent.py#L84)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `WorldAgentSettings` 创建后校验并规范化字段。 | `ValueError`, `isinstance`, `MapExtractionConfig`, `dict`, `SpatialPlannerConfig` |
| [`WorldAgentSettings.from_dict`](../saddlellm/WorldAgent.py#L95)<br><sub>`from_dict(cls, data: Optional[Dict[str, Any]]) -> 'WorldAgentSettings'`</sub> | method | 从字典解析并创建 `WorldAgentSettings`。 | `dict`, `values.pop`, `fields`, `sorted`, `set`, `ValueError`, `join`, `cls` |
| [`WorldAgentSettings.from_file`](../saddlellm/WorldAgent.py#L108)<br><sub>`from_file(cls, path: Union[str, os.PathLike]) -> 'WorldAgentSettings'`</sub> | method | `WorldAgentSettings` 中实现`from_file`的公开操作。 | `Path`, `source.is_file`, `FileNotFoundError`, `source.read_text`, `source.suffix.lower`, `json.loads`, `ImportError`, `yaml.safe_load`, `isinstance`, `ValueError` |
| [`WorldAgentSettings.to_dict`](../saddlellm/WorldAgent.py#L139)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldAgentSettings` 转为可序列化字典。 | `asdict`, `result.pop` |
| [`WorldObservation.from_dict`](../saddlellm/WorldAgent.py#L158)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'WorldObservation'`</sub> | method | 从字典解析并创建 `WorldObservation`。 | `cls`, `data.get`, `fields` |
| [`WorldObservation.to_dict`](../saddlellm/WorldAgent.py#L161)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldObservation` 转为可序列化字典。 | `asdict` |
| [`WorldState.to_dict`](../saddlellm/WorldAgent.py#L177)<br><sub>`to_dict(self, include_grid: bool=True) -> Dict[str, Any]`</sub> | method | 把 `WorldState` 转为可序列化字典。 | `self.observation.to_dict`, `_grid_to_dict`, `self.analysis.to_dict`, `dict`, `list` |
| [`WorldState.from_dict`](../saddlellm/WorldAgent.py#L191)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'WorldState'`</sub> | method | 从字典解析并创建 `WorldState`。 | `cls`, `str`, `data.get`, `WorldObservation.from_dict`, `_grid_from_dict`, `SpatialAnalysis.from_dict`, `dict`, `list`, `_utc_now` |
| [`WorldPlan.to_dict`](../saddlellm/WorldAgent.py#L221)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldPlan` 转为可序列化字典。 | `list` |
| [`WorldPlan.from_dict`](../saddlellm/WorldAgent.py#L239)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'WorldPlan'`</sub> | method | 从字典解析并创建 `WorldPlan`。 | `cls`, `str`, `data.get`, `_grid_point`, `list`, `dict`, `_utc_now` |
| [`WorldSimulation.to_dict`](../saddlellm/WorldAgent.py#L271)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldSimulation` 转为可序列化字典。 | `asdict` |
| [`WorldSimulation.from_dict`](../saddlellm/WorldAgent.py#L275)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'WorldSimulation'`</sub> | method | 从字典解析并创建 `WorldSimulation`。 | `fields`, `cls`, `data.items` |
| [`WorldFeedback.to_dict`](../saddlellm/WorldAgent.py#L296)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldFeedback` 转为可序列化字典。 | `asdict`, `list` |
| [`WorldFeedback.from_dict`](../saddlellm/WorldAgent.py#L302)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'WorldFeedback'`</sub> | method | 从字典解析并创建 `WorldFeedback`。 | `dict`, `_grid_point`, `values.get`, `fields`, `cls`, `values.items` |
| [`WorldAgentStore.__init__`](../saddlellm/WorldAgent.py#L319)<br><sub>`__init__(self, workspace: Union[str, os.PathLike]) -> None`</sub> | method | 初始化 `WorldAgentStore` 实例及其运行依赖。 | `resolve`, `Path`, `self.workspace.mkdir`, `mkdir`, `threading.RLock` |
| [`WorldAgentStore.save`](../saddlellm/WorldAgent.py#L328)<br><sub>`save(self, collection: str, entity_id: str, payload: Dict[str, Any]) -> Path`</sub> | method | `WorldAgentStore` 中保存`save`的公开操作。 | `ValueError`, `_entity_id`, `destination.with_suffix`, `json.dumps`, `temporary.write_text`, `temporary.replace`, `self._append_jsonl`, `collection.rstrip`, `payload.get`, `_utc_now` |
| [`WorldAgentStore.load`](../saddlellm/WorldAgent.py#L349)<br><sub>`load(self, collection: str, entity_id: str) -> Dict[str, Any]`</sub> | method | `WorldAgentStore` 中加载`load`的公开操作。 | `ValueError`, `_entity_id`, `path.is_file`, `WorldAgentError`, `collection.rstrip`, `json.loads`, `path.read_text` |
| [`WorldAgentStore.list`](../saddlellm/WorldAgent.py#L362)<br><sub>`list(self, collection: str, limit: int=50) -> List[Dict[str, Any]]`</sub> | method | `WorldAgentStore` 中列出`list`的公开操作。 | `ValueError`, `max`, `min`, `int`, `sorted`, `glob`, `path.stat`, `json.loads`, `path.read_text` |
| [`WorldAgentStore.append_replay`](../saddlellm/WorldAgent.py#L373)<br><sub>`append_replay(self, record: Dict[str, Any]) -> None`</sub> | method | `WorldAgentStore` 中追加`append_replay`的公开操作。 | `self._append_jsonl` |
| [`WorldAgentStore.summary`](../saddlellm/WorldAgent.py#L377)<br><sub>`summary(self) -> Dict[str, Any]`</sub> | method | `WorldAgentStore` 中实现`summary`的公开操作。 | `sum`, `glob`, `str`, `get`, `json.loads`, `path.read_text`, `outcomes.get`, `self.replay_path.is_file`, `self.replay_path.open`, `line.strip` |
| [`WorldAgentStore._append_jsonl`](../saddlellm/WorldAgent.py#L402)<br><sub>`_append_jsonl(path: Path, payload: Dict[str, Any]) -> None`</sub> | method | `WorldAgentStore` 中追加`append_jsonl`的内部辅助逻辑。 | `path.open`, `handle.write`, `json.dumps` |
| [`WorldAgentRuntime.__init__`](../saddlellm/WorldAgent.py#L410)<br><sub>`__init__(self, settings: Optional[WorldAgentSettings]=None, store: Optional[WorldAgentStore]=None, analyzer_loader: Optional[Callable[[], Any]]=None, world_model_loader: Optional[Callable[[], Any]]=None) -> None`</sub> | method | 初始化 `WorldAgentRuntime` 实例及其运行依赖。 | `WorldAgentSettings`, `WorldAgentStore`, `threading.Lock`, `SpatialObservationEncoder` |
| [`WorldAgentRuntime.capabilities`](../saddlellm/WorldAgent.py#L427)<br><sub>`capabilities(self) -> Dict[str, Any]`</sub> | method | `WorldAgentRuntime` 中实现`capabilities`的公开操作。 | `bool`, `_display_name`, `self.store.summary` |
| [`WorldAgentRuntime._world_model_available`](../saddlellm/WorldAgent.py#L460)<br><sub>`_world_model_available(self) -> bool`</sub> | method | `WorldAgentRuntime` 中实现世界、模型的内部辅助逻辑。 | `bool` |
| [`WorldAgentRuntime.analyze_image`](../saddlellm/WorldAgent.py#L463)<br><sub>`analyze_image(self, image_path: Union[str, os.PathLike], instruction: str='', semantic_backend: Literal['disabled', 'qwen-vl']='disabled', map_config: Optional[Union[MapExtractionConfig, Dict[str, Any]]]=None, observation_id: Optional[str]=None, source_name: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> WorldState`</sub> | method | `WorldAgentRuntime` 中分析图像的公开操作。 | `resolve`, `Path`, `path.is_file`, `WorldAgentError`, `_coerce_map_config`, `extract`, `TopDownMapExtractor`, `str`, `self._get_analyzer`, `analyzer.analyze` |
| [`WorldAgentRuntime.analyze_bytes`](../saddlellm/WorldAgent.py#L545)<br><sub>`analyze_bytes(self, payload: bytes, instruction: str='', semantic_backend: Literal['disabled', 'qwen-vl']='disabled', map_config: Optional[Union[MapExtractionConfig, Dict[str, Any]]]=None, source_name: str='upload') -> WorldState`</sub> | method | `WorldAgentRuntime` 中分析`analyze_bytes`的公开操作。 | `len`, `WorldAgentError`, `uuid.uuid4`, `destination.with_suffix`, `Image.open`, `io.BytesIO`, `save`, `source.convert`, `temporary.replace`, `temporary.unlink` |
| [`WorldAgentRuntime.plan`](../saddlellm/WorldAgent.py#L602)<br><sub>`plan(self, state_id: str, start: PointLike, goal: PointLike, instruction: str='', route_count: Optional[int]=None, use_world_model: bool=False, planner_config: Optional[Union[SpatialPlannerConfig, Dict[str, Any]]]=None, allow_perspective: Optional[bool]=None, language: str='zh-CN') -> WorldPlan`</sub> | method | `WorldAgentRuntime` 中规划计划的公开操作。 | `self.get_state`, `int`, `WorldAgentError`, `_coerce_planner_config`, `replace`, `GridPathPlanner`, `bool`, `SpatialWorldModelCoordinator`, `coordinator.plan_grid`, `str` |
| [`WorldAgentRuntime.simulate`](../saddlellm/WorldAgent.py#L716)<br><sub>`simulate(self, plan_id: str, route_id: Optional[str]=None, mode: Literal['auto', 'geometry', 'world_model']='auto', max_world_model_steps: int=64) -> WorldSimulation`</sub> | method | `WorldAgentRuntime` 中模拟`simulate`的公开操作。 | `WorldAgentError`, `self.get_plan`, `self.get_state`, `str`, `_find_route`, `_simulate_geometry`, `self._simulate_learned`, `int`, `isinstance`, `warnings.append` |
| [`WorldAgentRuntime.record_feedback`](../saddlellm/WorldAgent.py#L793)<br><sub>`record_feedback(self, plan_id: str, outcome: Literal['success', 'collision', 'blocked', 'cancelled', 'unknown'], route_id: Optional[str]=None, actual_path: Optional[Sequence[Sequence[float]]]=None, coordinate_space: Literal['grid', 'source']='grid', simulation_id: Optional[str]=None, note: str='', metrics: Optional[Dict[str, Any]]=None) -> WorldFeedback`</sub> | method | `WorldAgentRuntime` 中记录反馈的公开操作。 | `WorldAgentError`, `self.get_plan`, `self.get_state`, `str`, `_find_route`, `self.get_simulation`, `state.grid.scale_from_source`, `_grid_point`, `state.grid.in_bounds`, `actual.append` |
| [`WorldAgentRuntime.run_image`](../saddlellm/WorldAgent.py#L872)<br><sub>`run_image(self, image_path: Union[str, os.PathLike], start: PointLike, goal: PointLike, instruction: str='', semantic_backend: Literal['disabled', 'qwen-vl']='disabled', route_count: int=3, use_world_model: bool=False, simulation_mode: Literal['auto', 'geometry', 'world_model']='auto') -> Dict[str, Any]`</sub> | method | `WorldAgentRuntime` 中执行图像的公开操作。 | `self.analyze_image`, `self.plan`, `self.simulate`, `state.to_dict`, `plan.to_dict`, `simulation.to_dict`, `self.memory_summary` |
| [`WorldAgentRuntime.get_state`](../saddlellm/WorldAgent.py#L904)<br><sub>`get_state(self, state_id: str) -> WorldState`</sub> | method | `WorldAgentRuntime` 中读取状态的公开操作。 | `WorldState.from_dict`, `self.store.load` |
| [`WorldAgentRuntime.get_plan`](../saddlellm/WorldAgent.py#L907)<br><sub>`get_plan(self, plan_id: str) -> WorldPlan`</sub> | method | `WorldAgentRuntime` 中读取计划的公开操作。 | `WorldPlan.from_dict`, `self.store.load` |
| [`WorldAgentRuntime.get_simulation`](../saddlellm/WorldAgent.py#L910)<br><sub>`get_simulation(self, simulation_id: str) -> WorldSimulation`</sub> | method | `WorldAgentRuntime` 中读取`get_simulation`的公开操作。 | `WorldSimulation.from_dict`, `self.store.load` |
| [`WorldAgentRuntime.get_feedback`](../saddlellm/WorldAgent.py#L913)<br><sub>`get_feedback(self, feedback_id: str) -> WorldFeedback`</sub> | method | `WorldAgentRuntime` 中读取反馈的公开操作。 | `WorldFeedback.from_dict`, `self.store.load` |
| [`WorldAgentRuntime.memory_summary`](../saddlellm/WorldAgent.py#L916)<br><sub>`memory_summary(self) -> Dict[str, Any]`</sub> | method | `WorldAgentRuntime` 中实现`memory_summary`的公开操作。 | `self.store.summary` |
| [`WorldAgentRuntime._get_analyzer`](../saddlellm/WorldAgent.py#L919)<br><sub>`_get_analyzer(self) -> Any`</sub> | method | `WorldAgentRuntime` 中读取`get_analyzer`的内部辅助逻辑。 | `WorldAgentError`, `self._analyzer_loader`, `QwenVLSpatialAnalyzer.from_pretrained`, `str` |
| [`WorldAgentRuntime._get_world_model`](../saddlellm/WorldAgent.py#L938)<br><sub>`_get_world_model(self) -> Any`</sub> | method | `WorldAgentRuntime` 中读取世界、模型的内部辅助逻辑。 | `WorldAgentError`, `self._world_model_loader`, `WorldModelRuntime.from_pretrained`, `str` |
| [`WorldAgentRuntime._encode_for_world_model`](../saddlellm/WorldAgent.py#L958)<br><sub>`_encode_for_world_model(self, runtime: Any, grid: OccupancyGrid, start: GridPoint, goal: GridPoint) -> Tuple[np.ndarray, str]`</sub> | method | `WorldAgentRuntime` 中编码世界、模型的内部辅助逻辑。 | `tuple`, `int`, `len`, `self.observation_encoder.encode`, `clearance_map`, `GridPathPlanner`, `WorldAgentError`, `SpatialTensorObservationEncoder`, `min`, `max` |
| [`WorldAgentRuntime._simulate_learned`](../saddlellm/WorldAgent.py#L996)<br><sub>`_simulate_learned(self, state: WorldState, plan: WorldPlan, route: Dict[str, Any], max_steps: int) -> Dict[str, Any]`</sub> | method | `WorldAgentRuntime` 中模拟`simulate_learned`的内部辅助逻辑。 | `self._get_world_model`, `WorldAgentError`, `self._encode_for_world_model`, `runtime.encode_observation`, `_route_actions`, `_grid_point`, `int`, `runtime.rollout`, `_batch_vector`, `np.power` |
| [`WorldAgentRuntime._feedback_replay_record`](../saddlellm/WorldAgent.py#L1054)<br><sub>`_feedback_replay_record(self, feedback_id: str, state: WorldState, plan: WorldPlan, route_id: str, outcome: str, actual_path: Sequence[GridPoint]) -> Dict[str, Any]`</sub> | method | `WorldAgentRuntime` 中记录反馈的内部辅助逻辑。 | `clearance_map`, `GridPathPlanner`, `tolist`, `self.observation_encoder.encode`, `enumerate`, `zip`, `float`, `max`, `math.hypot`, `actions.append` |
| [`_grid_to_dict`](../saddlellm/WorldAgent.py#L1114)<br><sub>`_grid_to_dict(grid: OccupancyGrid, include_grid: bool) -> Dict[str, Any]`</sub> | function | 模块级实现占用栅格的内部辅助逻辑。 | `grid.to_dict`, `_rle_encode`, `grid.cells.reshape` |
| [`_grid_from_dict`](../saddlellm/WorldAgent.py#L1122)<br><sub>`_grid_from_dict(data: Dict[str, Any]) -> OccupancyGrid`</sub> | function | 模块级实现占用栅格的内部辅助逻辑。 | `np.asarray`, `data.get`, `_rle_decode`, `int`, `flat.reshape`, `ValueError`, `OccupancyGrid`, `float`, `tuple` |
| [`_rle_encode`](../saddlellm/WorldAgent.py#L1141)<br><sub>`_rle_encode(values: np.ndarray) -> List[List[int]]`</sub> | function | 模块级编码`rle_encode`的内部辅助逻辑。 | `reshape`, `np.asarray`, `int`, `runs.append` |
| [`_rle_decode`](../saddlellm/WorldAgent.py#L1159)<br><sub>`_rle_decode(runs: Iterable[Sequence[int]], expected: int) -> np.ndarray`</sub> | function | 模块级解码`rle_decode`的内部辅助逻辑。 | `len`, `ValueError`, `int`, `chunks.append`, `np.full`, `np.concatenate`, `np.empty` |
| [`_simulate_geometry`](../saddlellm/WorldAgent.py#L1177)<br><sub>`_simulate_geometry(grid: OccupancyGrid, route: Dict[str, Any], goal: GridPoint) -> Dict[str, Any]`</sub> | function | 模块级模拟`simulate_geometry`的内部辅助逻辑。 | `_grid_point`, `route.get`, `WorldAgentError`, `enumerate`, `grid.in_bounds`, `grid.traversable`, `math.hypot`, `int`, `steps.append`, `list` |
| [`_route_actions`](../saddlellm/WorldAgent.py#L1229)<br><sub>`_route_actions(points: Sequence[GridPoint], action_dim: int, max_steps: int) -> Tuple[np.ndarray, List[GridPoint]]`</sub> | function | 模块级实现路线的内部辅助逻辑。 | `len`, `WorldAgentError`, `astype`, `round`, `np.linspace`, `int`, `list`, `np.zeros`, `enumerate`, `zip` |
| [`_route_explanation`](../saddlellm/WorldAgent.py#L1251)<br><sub>`_route_explanation(route: Dict[str, Any], route_count: int, ranking_mode: str, language: str) -> Dict[str, Any]`</sub> | function | 模块级实现路线的内部辅助逻辑。 | `float`, `route.get`, `int`, `startswith`, `lower`, `str` |
| [`_compare_paths`](../saddlellm/WorldAgent.py#L1283)<br><sub>`_compare_paths(planned: Sequence[GridPoint], actual: Sequence[GridPoint], goal: GridPoint) -> Dict[str, Any]`</sub> | function | 模块级比较`compare_paths`的内部辅助逻辑。 | `set`, `float`, `math.dist`, `len`, `max` |
| [`_find_route`](../saddlellm/WorldAgent.py#L1310)<br><sub>`_find_route(plan: WorldPlan, route_id: str) -> Dict[str, Any]`</sub> | function | 模块级查找路线的内部辅助逻辑。 | `str`, `route.get`, `WorldAgentError` |
| [`_batch_vector`](../saddlellm/WorldAgent.py#L1317)<br><sub>`_batch_vector(value: Any) -> np.ndarray`</sub> | function | 模块级实现批次的内部辅助逻辑。 | `numpy`, `cpu`, `value.detach`, `ValueError`, `reshape`, `np.asarray` |
| [`_batch_step_means`](../saddlellm/WorldAgent.py#L1324)<br><sub>`_batch_step_means(value: Any) -> np.ndarray`</sub> | function | 模块级实现批次的内部辅助逻辑。 | `numpy`, `cpu`, `value.detach`, `ValueError`, `mean`, `reshape`, `np.asarray` |
| [`_summarize_occupancy`](../saddlellm/WorldAgent.py#L1332)<br><sub>`_summarize_occupancy(logits: Any) -> Dict[str, Any]`</sub> | function | 模块级汇总`summarize_occupancy`的内部辅助逻辑。 | `softmax`, `cpu`, `logits.detach`, `numpy`, `probabilities.argmax`, `probabilities.max`, `enumerate`, `zip`, `np.unique`, `summaries.append` |
| [`_coerce_map_config`](../saddlellm/WorldAgent.py#L1349)<br><sub>`_coerce_map_config(value: Union[MapExtractionConfig, Dict[str, Any]]) -> MapExtractionConfig`</sub> | function | 模块级实现配置的内部辅助逻辑。 | `isinstance`, `MapExtractionConfig`, `dict` |
| [`_coerce_planner_config`](../saddlellm/WorldAgent.py#L1353)<br><sub>`_coerce_planner_config(value: Union[SpatialPlannerConfig, Dict[str, Any]]) -> SpatialPlannerConfig`</sub> | function | 模块级实现配置的内部辅助逻辑。 | `isinstance`, `SpatialPlannerConfig`, `dict` |
| [`_grid_point`](../saddlellm/WorldAgent.py#L1359)<br><sub>`_grid_point(value: Sequence[Any]) -> GridPoint`</sub> | function | 模块级实现占用栅格的内部辅助逻辑。 | `len`, `WorldAgentError`, `int`, `round`, `float` |
| [`_normalized_goal_distance`](../saddlellm/WorldAgent.py#L1365)<br><sub>`_normalized_goal_distance(grid: OccupancyGrid, point: GridPoint, goal: GridPoint) -> float`</sub> | function | 模块级实现`normalized_goal_distance`的内部辅助逻辑。 | `max`, `math.hypot` |
| [`_entity_id`](../saddlellm/WorldAgent.py#L1375)<br><sub>`_entity_id(value: str) -> str`</sub> | function | 模块级实现`entity_id`的内部辅助逻辑。 | `lower`, `strip`, `str`, `len`, `any`, `WorldAgentError` |
| [`_sha256`](../saddlellm/WorldAgent.py#L1382)<br><sub>`_sha256(path: Path) -> str`</sub> | function | 模块级实现`sha256`的内部辅助逻辑。 | `hashlib.sha256`, `path.open`, `iter`, `handle.read`, `digest.update`, `digest.hexdigest` |
| [`_display_name`](../saddlellm/WorldAgent.py#L1390)<br><sub>`_display_name(path: Optional[str]) -> Optional[str]`</sub> | function | 模块级实现`display_name`的内部辅助逻辑。 | `rstrip`, `str`, `os.path.basename` |
| [`_clamp`](../saddlellm/WorldAgent.py#L1397)<br><sub>`_clamp(value: float) -> float`</sub> | function | 模块级实现`clamp`的内部辅助逻辑。 | `min`, `max`, `float` |
| [`_utc_now`](../saddlellm/WorldAgent.py#L1401)<br><sub>`_utc_now() -> str`</sub> | function | 模块级实现`utc_now`的内部辅助逻辑。 | `replace`, `isoformat`, `datetime.now` |

## `saddlellm/WorldAgentAPI.py`

共 21 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`WorldAnalyzeRequest.map_config`](../saddlellm/WorldAgentAPI.py#L39)<br><sub>`map_config(self, base: MapExtractionConfig) -> MapExtractionConfig`</sub> | method | `WorldAnalyzeRequest` 中实现配置的公开操作。 | `asdict`, `getattr`, `MapExtractionConfig` |
| [`WorldPlanRequest.validate_endpoint`](../saddlellm/WorldAgentAPI.py#L67)<br><sub>`validate_endpoint(cls, value: Endpoint) -> Endpoint`</sub> | method | `WorldPlanRequest` 中校验`validate_endpoint`的公开操作。 | `isinstance`, `value.strip`, `ValueError`, `len`, `all`, `math.isfinite`, `float` |
| [`WorldPlanRequest.planner_config`](../saddlellm/WorldAgentAPI.py#L77)<br><sub>`planner_config(self, base: SpatialPlannerConfig) -> SpatialPlannerConfig`</sub> | method | `WorldPlanRequest` 中实现配置的公开操作。 | `asdict`, `getattr`, `SpatialPlannerConfig` |
| [`WorldFeedbackRequest.validate_actual_path`](../saddlellm/WorldAgentAPI.py#L114)<br><sub>`validate_actual_path(cls, value: List[Coordinate]) -> List[Coordinate]`</sub> | method | `WorldFeedbackRequest` 中校验路径的公开操作。 | `len`, `all`, `math.isfinite`, `float`, `ValueError` |
| [`_world_agent_error_handler`](../saddlellm/WorldAgentAPI.py#L121)<br><sub>`async _world_agent_error_handler(_request: Any, error: WorldAgentError) -> JSONResponse`</sub> | function | 模块级实现世界的内部辅助逻辑。 | `JSONResponse`, `str` |
| [`create_world_agent_router`](../saddlellm/WorldAgentAPI.py#L131)<br><sub>`create_world_agent_router(runtime: WorldAgentRuntime, prefix: str='/v1/world') -> APIRouter`</sub> | function | 模块级创建世界的公开操作。 | `APIRouter` |
| [`create_world_agent_router.health`](../saddlellm/WorldAgentAPI.py#L138)<br><sub>`async health() -> Dict[str, Any]`</sub> | nested function | 模块级实现`health`的局部回调/辅助逻辑。 | — |
| [`create_world_agent_router.capabilities`](../saddlellm/WorldAgentAPI.py#L146)<br><sub>`async capabilities() -> Dict[str, Any]`</sub> | nested function | 模块级实现`capabilities`的局部回调/辅助逻辑。 | `runtime.capabilities` |
| [`create_world_agent_router.memory`](../saddlellm/WorldAgentAPI.py#L150)<br><sub>`async memory() -> Dict[str, Any]`</sub> | nested function | 模块级实现`memory`的局部回调/辅助逻辑。 | `runtime.memory_summary` |
| [`create_world_agent_router.analyze`](../saddlellm/WorldAgentAPI.py#L154)<br><sub>`async analyze(image: UploadFile=File(...), request: str=Form('{}')) -> Dict[str, Any]`</sub> | nested function | 模块级分析`analyze`的局部回调/辅助逻辑。 | `WorldAnalyzeRequest.model_validate_json`, `WorldAgentError`, `image.read`, `run_in_threadpool`, `parsed.map_config`, `state.to_dict` |
| [`create_world_agent_router.plan`](../saddlellm/WorldAgentAPI.py#L175)<br><sub>`async plan(request: WorldPlanRequest) -> Dict[str, Any]`</sub> | nested function | 模块级规划计划的局部回调/辅助逻辑。 | `run_in_threadpool`, `request.planner_config`, `result.to_dict` |
| [`create_world_agent_router.simulate`](../saddlellm/WorldAgentAPI.py#L191)<br><sub>`async simulate(request: WorldSimulationRequest) -> Dict[str, Any]`</sub> | nested function | 模块级模拟`simulate`的局部回调/辅助逻辑。 | `run_in_threadpool`, `result.to_dict` |
| [`create_world_agent_router.feedback`](../saddlellm/WorldAgentAPI.py#L202)<br><sub>`async feedback(request: WorldFeedbackRequest) -> Dict[str, Any]`</sub> | nested function | 模块级实现反馈的局部回调/辅助逻辑。 | `run_in_threadpool`, `result.to_dict`, `runtime.memory_summary` |
| [`create_world_agent_router.get_state`](../saddlellm/WorldAgentAPI.py#L221)<br><sub>`async get_state(state_id: str) -> Dict[str, Any]`</sub> | nested function | 模块级读取状态的局部回调/辅助逻辑。 | `to_dict`, `runtime.get_state` |
| [`create_world_agent_router.get_plan`](../saddlellm/WorldAgentAPI.py#L225)<br><sub>`async get_plan(plan_id: str) -> Dict[str, Any]`</sub> | nested function | 模块级读取计划的局部回调/辅助逻辑。 | `to_dict`, `runtime.get_plan` |
| [`create_world_agent_router.get_simulation`](../saddlellm/WorldAgentAPI.py#L229)<br><sub>`async get_simulation(simulation_id: str) -> Dict[str, Any]`</sub> | nested function | 模块级读取`get_simulation`的局部回调/辅助逻辑。 | `to_dict`, `runtime.get_simulation` |
| [`create_world_agent_router.get_feedback`](../saddlellm/WorldAgentAPI.py#L233)<br><sub>`async get_feedback(feedback_id: str) -> Dict[str, Any]`</sub> | nested function | 模块级读取反馈的局部回调/辅助逻辑。 | `to_dict`, `runtime.get_feedback` |
| [`create_world_agent_router.list_records`](../saddlellm/WorldAgentAPI.py#L237)<br><sub>`async list_records(collection: str, limit: int=50) -> Dict[str, Any]`</sub> | nested function | 模块级列出`list_records`的局部回调/辅助逻辑。 | `WorldAgentError`, `runtime.store.list` |
| [`install_world_agent_routes`](../saddlellm/WorldAgentAPI.py#L248)<br><sub>`install_world_agent_routes(app: FastAPI, runtime: WorldAgentRuntime, prefix: str='/v1/world') -> None`</sub> | function | 模块级安装世界的公开操作。 | `app.add_exception_handler`, `app.include_router`, `create_world_agent_router` |
| [`create_world_agent_app`](../saddlellm/WorldAgentAPI.py#L258)<br><sub>`create_world_agent_app(settings: Optional[WorldAgentSettings]=None, runtime: Optional[WorldAgentRuntime]=None) -> FastAPI`</sub> | function | 模块级创建世界的公开操作。 | `WorldAgentSettings`, `WorldAgentRuntime`, `FastAPI`, `app.add_middleware`, `install_world_agent_routes` |
| [`create_world_agent_app.root`](../saddlellm/WorldAgentAPI.py#L280)<br><sub>`async root() -> Dict[str, Any]`</sub> | nested function | 模块级实现`root`的局部回调/辅助逻辑。 | — |

## `saddlellm/WorldModelBackends.py`

共 4 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`normalize_world_model_backend`](../saddlellm/WorldModelBackends.py#L44)<br><sub>`normalize_world_model_backend(name: Optional[str]) -> str`</sub> | function | 模块级规范化世界、模型、后端的公开操作。 | `lower`, `strip`, `str`, `aliases.get` |
| [`create_world_model`](../saddlellm/WorldModelBackends.py#L57)<br><sub>`create_world_model(backend: str, config: Dict[str, Any])`</sub> | function | Instantiate a native model from a backend name and inferred config. | `normalize_world_model_backend`, `dict`, `values.pop`, `WorldModel`, `WorldModelConfig.from_dict`, `CategoricalWorldModel`, `CategoricalWorldModelConfig.from_dict`, `ValueError`, `join`, `sorted` |
| [`load_world_model`](../saddlellm/WorldModelBackends.py#L73)<br><sub>`load_world_model(path: str, map_location: Any='cpu')`</sub> | function | Load any native SaddleLLM world model from one output directory. | `os.path.join`, `open`, `json.load`, `normalize_world_model_backend`, `config.get`, `WorldModel.from_pretrained`, `CategoricalWorldModel.from_pretrained`, `ValueError` |
| [`list_world_model_backends`](../saddlellm/WorldModelBackends.py#L87)<br><sub>`list_world_model_backends() -> List[Dict[str, Any]]`</sub> | function | List the world-model implementations owned and trained by SaddleLLM. | `dict`, `NATIVE_WORLD_MODEL_BACKENDS.values` |

## `saddlellm/WorldModelData.py`

共 22 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`load_world_model_trajectories`](../saddlellm/WorldModelData.py#L20)<br><sub>`load_world_model_trajectories(path: str) -> List[Dict[str, Any]]`</sub> | function | Load JSON/JSONL, NPZ, or PyTorch trajectory data. | `os.path.isfile`, `FileNotFoundError`, `lower`, `os.path.splitext`, `open`, `enumerate`, `line.strip`, `records.append`, `json.loads`, `WorldModelDataError` |
| [`normalize_world_model_trajectories`](../saddlellm/WorldModelData.py#L64)<br><sub>`normalize_world_model_trajectories(payload: Any) -> List[Dict[str, Any]]`</sub> | function | Normalize trajectory records or transition rows to one canonical schema. | `isinstance`, `_first_present`, `list`, `WorldModelDataError`, `type`, `all`, `enumerate`, `_looks_like_transition`, `transitions.append`, `trajectories.append` |
| [`split_world_model_trajectories`](../saddlellm/WorldModelData.py#L99)<br><sub>`split_world_model_trajectories(trajectories: Sequence[Dict[str, Any]], validation_split: float, seed: int=42, group_key: Optional[str]=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`</sub> | function | Split complete episodes, optionally holding out whole scene groups. | `list`, `len`, `ValueError`, `set`, `OrderedDict`, `enumerate`, `_nested_metadata_value`, `item.get`, `WorldModelDataError`, `append` |
| [`infer_world_model_dimensions`](../saddlellm/WorldModelData.py#L153)<br><sub>`infer_world_model_dimensions(trajectories: Sequence[Dict[str, Any]], action_type: str='continuous') -> Dict[str, Any]`</sub> | function | Infer observation shape and continuous width/discrete vocabulary size. | `WorldModelDataError`, `tuple`, `np.asarray`, `enumerate`, `lower`, `str`, `np.min`, `max`, `int`, `np.max` |
| [`WorldModelTrajectoryDataset.__init__`](../saddlellm/WorldModelData.py#L211)<br><sub>`__init__(self, trajectories: Sequence[Dict[str, Any]], sequence_length: int=32, stride: Optional[int]=None, action_type: str='continuous', action_dim: Optional[int]=None, normalize_images: bool=True, pad_short_trajectories: bool=True) -> None`</sub> | method | 初始化 `WorldModelTrajectoryDataset` 实例及其运行依赖。 | `ValueError`, `normalize_world_model_trajectories`, `list`, `WorldModelDataError`, `int`, `lower`, `str`, `bool`, `infer_world_model_dimensions`, `self._build_windows` |
| [`WorldModelTrajectoryDataset.from_file`](../saddlellm/WorldModelData.py#L256)<br><sub>`from_file(cls, path: str, sequence_length: int=32, stride: Optional[int]=None, action_type: str='continuous', action_dim: Optional[int]=None, normalize_images: bool=True, pad_short_trajectories: bool=True) -> 'WorldModelTrajectoryDataset'`</sub> | method | `WorldModelTrajectoryDataset` 中实现`from_file`的公开操作。 | `cls`, `load_world_model_trajectories` |
| [`WorldModelTrajectoryDataset.__len__`](../saddlellm/WorldModelData.py#L276)<br><sub>`__len__(self) -> int`</sub> | method | `WorldModelTrajectoryDataset` 中实现`len__`的内部辅助逻辑。 | `len` |
| [`WorldModelTrajectoryDataset.__getitem__`](../saddlellm/WorldModelData.py#L279)<br><sub>`__getitem__(self, index: int) -> Dict[str, Any]`</sub> | method | `WorldModelTrajectoryDataset` 中实现`getitem__`的内部辅助逻辑。 | `self._observation_tensor`, `self._action_tensor`, `reshape`, `torch.as_tensor`, `expand`, `torch.cat`, `torch.zeros`, `torch.ones`, `list`, `auxiliary_windows.items` |
| [`WorldModelTrajectoryDataset.summary`](../saddlellm/WorldModelData.py#L335)<br><sub>`summary(self) -> Dict[str, Any]`</sub> | method | `WorldModelTrajectoryDataset` 中实现`summary`的公开操作。 | `sum`, `len`, `int`, `list`, `all` |
| [`WorldModelTrajectoryDataset._build_windows`](../saddlellm/WorldModelData.py#L355)<br><sub>`_build_windows(self) -> List[Tuple[int, int, int]]`</sub> | method | `WorldModelTrajectoryDataset` 中构建`build_windows`的内部辅助逻辑。 | `enumerate`, `len`, `windows.append`, `list`, `range`, `starts.append`, `windows.extend` |
| [`WorldModelTrajectoryDataset._observation_tensor`](../saddlellm/WorldModelData.py#L374)<br><sub>`_observation_tensor(self, values: Any) -> torch.Tensor`</sub> | method | `WorldModelTrajectoryDataset` 中实现观测的内部辅助逻辑。 | `_as_numpy`, `torch.as_tensor`, `WorldModelDataError`, `len`, `np.issubdtype`, `tensor.numel`, `float`, `max`, `tensor.detach` |
| [`WorldModelTrajectoryDataset._action_tensor`](../saddlellm/WorldModelData.py#L389)<br><sub>`_action_tensor(self, values: Any) -> torch.Tensor`</sub> | method | `WorldModelTrajectoryDataset` 中实现动作的内部辅助逻辑。 | `_as_numpy`, `torch.as_tensor`, `WorldModelDataError`, `tensor.unsqueeze`, `tensor.flatten` |
| [`_canonicalize_trajectory`](../saddlellm/WorldModelData.py#L404)<br><sub>`_canonicalize_trajectory(record: Dict[str, Any], index: int) -> Dict[str, Any]`</sub> | function | 模块级实现`canonicalize_trajectory`的内部辅助逻辑。 | `_first_present`, `WorldModelDataError`, `_as_numpy`, `actions_array.reshape`, `int`, `np.concatenate`, `_transition_vector`, `record.get`, `_transition_matrix` |
| [`_transitions_to_trajectories`](../saddlellm/WorldModelData.py#L460)<br><sub>`_transitions_to_trajectories(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]`</sub> | function | 模块级实现`transitions_to_trajectories`的内部辅助逻辑。 | `OrderedDict`, `enumerate`, `str`, `record.get`, `dict`, `append`, `groups.setdefault`, `groups.items`, `rows.sort`, `row.get` |
| [`_transition_segment_to_trajectory`](../saddlellm/WorldModelData.py#L489)<br><sub>`_transition_segment_to_trajectory(episode_id: str, segment_index: int, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]`</sub> | function | 模块级实现`transition_segment_to_trajectory`的内部辅助逻辑。 | `_first_present`, `WorldModelDataError`, `observations.append`, `actions.append`, `rewards.append`, `float`, `dones.append`, `bool`, `_canonicalize_trajectory` |
| [`_looks_like_transition`](../saddlellm/WorldModelData.py#L522)<br><sub>`_looks_like_transition(record: Dict[str, Any]) -> bool`</sub> | function | 模块级实现`looks_like_transition`的内部辅助逻辑。 | `any` |
| [`_transition_vector`](../saddlellm/WorldModelData.py#L528)<br><sub>`_transition_vector(values: Any, length: int, default: float, name: str) -> np.ndarray`</sub> | function | 模块级实现`transition_vector`的内部辅助逻辑。 | `np.full`, `_as_numpy`, `WorldModelDataError`, `array.reshape`, `array.astype` |
| [`_transition_matrix`](../saddlellm/WorldModelData.py#L549)<br><sub>`_transition_matrix(values: Any, length: int, name: str) -> np.ndarray`</sub> | function | 模块级实现`transition_matrix`的内部辅助逻辑。 | `_as_numpy`, `WorldModelDataError`, `array.reshape`, `array.astype` |
| [`_nested_metadata_value`](../saddlellm/WorldModelData.py#L562)<br><sub>`_nested_metadata_value(metadata: Any, path: str) -> Any`</sub> | function | 模块级实现`nested_metadata_value`的内部辅助逻辑。 | `split`, `str`, `isinstance` |
| [`_first_present`](../saddlellm/WorldModelData.py#L571)<br><sub>`_first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any`</sub> | function | 模块级实现`first_present`的内部辅助逻辑。 | — |
| [`_as_numpy`](../saddlellm/WorldModelData.py#L578)<br><sub>`_as_numpy(value: Any) -> np.ndarray`</sub> | function | 模块级实现`as_numpy`的内部辅助逻辑。 | `isinstance`, `numpy`, `cpu`, `value.detach`, `np.asarray` |
| [`_torch_load_local_data`](../saddlellm/WorldModelData.py#L584)<br><sub>`_torch_load_local_data(path: str) -> Any`</sub> | function | 模块级加载数据的内部辅助逻辑。 | `torch.load` |

## `saddlellm/WorldModelInference.py`

共 25 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`WorldModelPlannerConfig.__post_init__`](../saddlellm/WorldModelInference.py#L32)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `WorldModelPlannerConfig` 创建后校验并规范化字段。 | `ValueError` |
| [`WorldModelPlannerConfig.to_dict`](../saddlellm/WorldModelInference.py#L48)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldModelPlannerConfig` 转为可序列化字典。 | `asdict` |
| [`WorldModelPlannerConfig.from_dict`](../saddlellm/WorldModelInference.py#L52)<br><sub>`from_dict(cls, data: Optional[Dict[str, Any]]) -> 'WorldModelPlannerConfig'`</sub> | method | 从字典解析并创建 `WorldModelPlannerConfig`。 | `dict`, `fields`, `sorted`, `set`, `ValueError`, `join`, `cls` |
| [`WorldModelPlan.action`](../saddlellm/WorldModelInference.py#L80)<br><sub>`action(self) -> torch.Tensor`</sub> | method | `WorldModelPlan` 中实现动作的公开操作。 | — |
| [`WorldModelPlan.to_dict`](../saddlellm/WorldModelInference.py#L83)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldModelPlan` 转为可序列化字典。 | `tolist`, `cpu`, `self.action.detach`, `self.actions.detach`, `float`, `self.score.detach`, `self.predicted_observations.detach`, `self.predicted_rewards.detach`, `self.predicted_continuation.detach` |
| [`WorldModelRuntime.__init__`](../saddlellm/WorldModelInference.py#L97)<br><sub>`__init__(self, model: torch.nn.Module, device: str='auto') -> None`</sub> | method | 初始化 `WorldModelRuntime` 实例及其运行依赖。 | `_resolve_device`, `model.to`, `self.model.eval` |
| [`WorldModelRuntime.from_pretrained`](../saddlellm/WorldModelInference.py#L103)<br><sub>`from_pretrained(cls, path: Union[str, os.PathLike], device: str='auto') -> 'WorldModelRuntime'`</sub> | method | 从检查点加载 `WorldModelRuntime`，遵循预训练模型的目录契约。 | `load_world_model`, `os.fspath`, `cls` |
| [`WorldModelRuntime.backend`](../saddlellm/WorldModelInference.py#L112)<br><sub>`backend(self) -> str`</sub> | method | `WorldModelRuntime` 中实现后端的公开操作。 | `getattr` |
| [`WorldModelRuntime.encode_observation`](../saddlellm/WorldModelInference.py#L116)<br><sub>`encode_observation(self, observation: Any, deterministic: bool=True) -> RSSMState`</sub> | method | Create a posterior belief from a current observation. | `self._observation_tensor`, `self.model.initial_state_from_observation` |
| [`WorldModelRuntime.filter`](../saddlellm/WorldModelInference.py#L130)<br><sub>`filter(self, observations: Any, actions: Any, deterministic: bool=True) -> Any`</sub> | method | Filter an observed history and return the model's posterior output. | `self._observation_tensor`, `len`, `observation_tensor.unsqueeze`, `ValueError`, `torch.as_tensor`, `action_tensor.unsqueeze`, `self.model` |
| [`WorldModelRuntime.rollout`](../saddlellm/WorldModelInference.py#L160)<br><sub>`rollout(self, state: RSSMState, actions: Any, deterministic: bool=True) -> WorldModelImagination`</sub> | method | Predict observations, rewards, and continuation without new observations. | `state.to`, `self._rollout_action_tensor`, `self.model.imagine` |
| [`WorldModelRuntime.predict`](../saddlellm/WorldModelInference.py#L177)<br><sub>`predict(self, observations: Any, history_actions: Any, future_actions: Any, deterministic: bool=True) -> WorldModelImagination`</sub> | method | Filter a history and immediately roll it forward under future actions. | `self.filter`, `self.rollout` |
| [`WorldModelRuntime.score_action_sequences`](../saddlellm/WorldModelInference.py#L198)<br><sub>`score_action_sequences(self, state: RSSMState, candidate_actions: Any, discount: float=0.99, continuation_aware: bool=True) -> WorldModelActionEvaluation`</sub> | method | Score candidates by discounted predicted reward in latent imagination. | `ValueError`, `self._candidate_action_tensor`, `self._expand_state`, `state.to`, `self.model.imagine`, `unsqueeze`, `torch.pow`, `imagination.rewards.new_tensor`, `torch.arange`, `torch.cumprod` |
| [`WorldModelRuntime.plan`](../saddlellm/WorldModelInference.py#L244)<br><sub>`plan(self, state: RSSMState, config: Optional[WorldModelPlannerConfig]=None) -> WorldModelPlan`</sub> | method | Choose an action sequence using native CEM/random-shooting planning. | `WorldModelPlannerConfig`, `state.to`, `ValueError`, `manual_seed`, `torch.Generator`, `self._plan_discrete`, `self._plan_continuous` |
| [`WorldModelRuntime._plan_continuous`](../saddlellm/WorldModelInference.py#L260)<br><sub>`_plan_continuous(self, state: RSSMState, config: WorldModelPlannerConfig, generator: torch.Generator) -> WorldModelPlan`</sub> | method | `WorldModelRuntime` 中规划计划的内部辅助逻辑。 | `_action_bound`, `torch.any`, `ValueError`, `repeat`, `max`, `int`, `range`, `torch.randn`, `mean.unsqueeze`, `std.unsqueeze` |
| [`WorldModelRuntime._plan_discrete`](../saddlellm/WorldModelInference.py#L302)<br><sub>`_plan_discrete(self, state: RSSMState, config: WorldModelPlannerConfig, generator: torch.Generator) -> WorldModelPlan`</sub> | method | `WorldModelRuntime` 中规划计划的内部辅助逻辑。 | `torch.full`, `max`, `int`, `range`, `transpose`, `torch.multinomial`, `candidates_cpu.to`, `self.score_action_sequences`, `self._keep_best`, `indices.cpu` |
| [`WorldModelRuntime._keep_best`](../saddlellm/WorldModelInference.py#L340)<br><sub>`_keep_best(current: Optional[Tuple[torch.Tensor, ...]], candidates: torch.Tensor, evaluation: WorldModelActionEvaluation) -> Tuple[torch.Tensor, ...]`</sub> | method | `WorldModelRuntime` 中实现`keep_best`的内部辅助逻辑。 | `int`, `evaluation.scores.argmax`, `float`, `clone`, `score.detach`, `detach` |
| [`WorldModelRuntime._build_plan`](../saddlellm/WorldModelInference.py#L359)<br><sub>`_build_plan(best: Optional[Tuple[torch.Tensor, ...]]) -> WorldModelPlan`</sub> | method | `WorldModelRuntime` 中构建计划的内部辅助逻辑。 | `RuntimeError`, `WorldModelPlan` |
| [`WorldModelRuntime._observation_tensor`](../saddlellm/WorldModelInference.py#L370)<br><sub>`_observation_tensor(self, value: Any) -> torch.Tensor`</sub> | method | `WorldModelRuntime` 中实现观测的内部辅助逻辑。 | `torch.as_tensor`, `tensor.to`, `next`, `self.model.parameters`, `tensor.numel`, `float`, `tensor.max` |
| [`WorldModelRuntime._rollout_action_tensor`](../saddlellm/WorldModelInference.py#L377)<br><sub>`_rollout_action_tensor(self, actions: Any, state_batch_size: int) -> torch.Tensor`</sub> | method | `WorldModelRuntime` 中实现动作的内部辅助逻辑。 | `torch.as_tensor`, `tensor.reshape`, `tensor.unsqueeze`, `ValueError` |
| [`WorldModelRuntime._candidate_action_tensor`](../saddlellm/WorldModelInference.py#L412)<br><sub>`_candidate_action_tensor(self, actions: Any) -> torch.Tensor`</sub> | method | `WorldModelRuntime` 中实现动作的内部辅助逻辑。 | `torch.as_tensor`, `tensor.unsqueeze`, `ValueError` |
| [`WorldModelRuntime._expand_state`](../saddlellm/WorldModelInference.py#L428)<br><sub>`_expand_state(state: RSSMState, batch_size: int) -> RSSMState`</sub> | method | `WorldModelRuntime` 中实现状态的内部辅助逻辑。 | `ValueError`, `RSSMState`, `contiguous`, `state.deterministic.expand`, `state.stochastic.expand`, `state.context.expand` |
| [`run_world_model_inference`](../saddlellm/WorldModelInference.py#L447)<br><sub>`run_world_model_inference(checkpoint: Union[str, os.PathLike], request: Dict[str, Any], device: str='auto') -> Dict[str, Any]`</sub> | function | Run JSON-compatible native rollout or planning against a checkpoint. | `WorldModelRuntime.from_pretrained`, `bool`, `request.get`, `ValueError`, `runtime.filter`, `runtime.encode_observation`, `os.path.abspath`, `os.fspath`, `runtime.rollout`, `tolist` |
| [`_action_bound`](../saddlellm/WorldModelInference.py#L496)<br><sub>`_action_bound(value: Union[float, Sequence[float]], action_dim: int, name: str) -> torch.Tensor`</sub> | function | 模块级实现动作的内部辅助逻辑。 | `flatten`, `torch.as_tensor`, `tensor.numel`, `tensor.repeat`, `ValueError`, `tensor.reshape` |
| [`_resolve_device`](../saddlellm/WorldModelInference.py#L509)<br><sub>`_resolve_device(requested: str) -> torch.device`</sub> | function | 模块级解析设备的内部辅助逻辑。 | `lower`, `str`, `torch.cuda.is_available`, `torch.device`, `hasattr`, `torch.backends.mps.is_available`, `RuntimeError` |

## `saddlellm/_CategoricalWorldModel.py`

共 37 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`symlog`](../saddlellm/_CategoricalWorldModel.py#L30)<br><sub>`symlog(value: torch.Tensor) -> torch.Tensor`</sub> | function | Signed logarithm for scale-robust regression. | `torch.sign`, `torch.log1p`, `torch.abs` |
| [`symexp`](../saddlellm/_CategoricalWorldModel.py#L36)<br><sub>`symexp(value: torch.Tensor) -> torch.Tensor`</sub> | function | Inverse of :func:`symlog`, with a safe exponent range. | `value.clamp`, `torch.sign`, `torch.expm1`, `torch.abs` |
| [`CategoricalWorldModelConfig.__post_init__`](../saddlellm/_CategoricalWorldModel.py#L72)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `CategoricalWorldModelConfig` 创建后校验并规范化字段。 | `tuple`, `int`, `lower`, `str`, `self._validate` |
| [`CategoricalWorldModelConfig.is_image`](../saddlellm/_CategoricalWorldModel.py#L80)<br><sub>`is_image(self) -> bool`</sub> | method | `CategoricalWorldModelConfig` 中实现图像的公开操作。 | `len` |
| [`CategoricalWorldModelConfig.categorical_size`](../saddlellm/_CategoricalWorldModel.py#L84)<br><sub>`categorical_size(self) -> int`</sub> | method | `CategoricalWorldModelConfig` 中实现`categorical_size`的公开操作。 | — |
| [`CategoricalWorldModelConfig.feature_size`](../saddlellm/_CategoricalWorldModel.py#L88)<br><sub>`feature_size(self) -> int`</sub> | method | `CategoricalWorldModelConfig` 中实现`feature_size`的公开操作。 | — |
| [`CategoricalWorldModelConfig.to_dict`](../saddlellm/_CategoricalWorldModel.py#L91)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `CategoricalWorldModelConfig` 转为可序列化字典。 | `asdict`, `list` |
| [`CategoricalWorldModelConfig.from_dict`](../saddlellm/_CategoricalWorldModel.py#L98)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'CategoricalWorldModelConfig'`</sub> | method | 从字典解析并创建 `CategoricalWorldModelConfig`。 | `dict`, `values.pop`, `fields`, `sorted`, `set`, `ValueError`, `join`, `cls` |
| [`CategoricalWorldModelConfig._validate`](../saddlellm/_CategoricalWorldModel.py#L109)<br><sub>`_validate(self) -> None`</sub> | method | `CategoricalWorldModelConfig` 中校验`validate`的内部辅助逻辑。 | `any`, `ValueError`, `getattr` |
| [`_RMSNorm.__init__`](../saddlellm/_CategoricalWorldModel.py#L165)<br><sub>`__init__(self, width: int, epsilon: float=1e-06) -> None`</sub> | method | 初始化 `_RMSNorm` 实例及其运行依赖。 | `__init__`, `super`, `nn.Parameter`, `torch.ones` |
| [`_RMSNorm.forward`](../saddlellm/_CategoricalWorldModel.py#L170)<br><sub>`forward(self, value: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `_RMSNorm` 的前向计算。 | `torch.rsqrt`, `mean`, `value.square` |
| [`_activation`](../saddlellm/_CategoricalWorldModel.py#L175)<br><sub>`_activation(name: str) -> nn.Module`</sub> | function | 模块级实现`activation`的内部辅助逻辑。 | `lower`, `str`, `nn.SiLU`, `nn.GELU`, `nn.ELU`, `nn.ReLU`, `ValueError` |
| [`_mlp`](../saddlellm/_CategoricalWorldModel.py#L188)<br><sub>`_mlp(input_size: int, output_size: int, hidden_size: int, layers: int, activation: str) -> nn.Sequential`</sub> | function | 模块级实现`mlp`的内部辅助逻辑。 | `range`, `modules.extend`, `nn.Linear`, `_RMSNorm`, `_activation`, `modules.append`, `nn.Sequential` |
| [`_GroupedLinear.__init__`](../saddlellm/_CategoricalWorldModel.py#L213)<br><sub>`__init__(self, groups: int, input_size: int, output_size: int) -> None`</sub> | method | 初始化 `_GroupedLinear` 实例及其运行依赖。 | `__init__`, `super`, `nn.Parameter`, `torch.empty`, `torch.zeros`, `nn.init.xavier_uniform_` |
| [`_GroupedLinear.forward`](../saddlellm/_CategoricalWorldModel.py#L222)<br><sub>`forward(self, value: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `_GroupedLinear` 的前向计算。 | `ValueError`, `tuple`, `torch.einsum` |
| [`_TwoHotSymlogHead.__init__`](../saddlellm/_CategoricalWorldModel.py#L232)<br><sub>`__init__(self, config: CategoricalWorldModelConfig) -> None`</sub> | method | 初始化 `_TwoHotSymlogHead` 实例及其运行依赖。 | `__init__`, `super`, `_mlp`, `self.register_buffer`, `torch.linspace` |
| [`_TwoHotSymlogHead.forward`](../saddlellm/_CategoricalWorldModel.py#L246)<br><sub>`forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`</sub> | method | 执行 `_TwoHotSymlogHead` 的前向计算。 | `self.network`, `sum`, `torch.softmax`, `symexp` |
| [`_TwoHotSymlogHead.loss`](../saddlellm/_CategoricalWorldModel.py#L251)<br><sub>`loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor`</sub> | method | `_TwoHotSymlogHead` 中实现`loss`的公开操作。 | `clamp`, `symlog`, `float`, `len`, `long`, `position.floor`, `lower.to`, `F.log_softmax`, `squeeze`, `torch.gather` |
| [`CategoricalRSSMCell.__init__`](../saddlellm/_CategoricalWorldModel.py#L274)<br><sub>`__init__(self, config: CategoricalWorldModelConfig) -> None`</sub> | method | 初始化 `CategoricalRSSMCell` 实例及其运行依赖。 | `__init__`, `super`, `nn.Sequential`, `nn.Linear`, `_RMSNorm`, `_activation`, `_GroupedLinear`, `_mlp` |
| [`CategoricalRSSMCell.initial_posterior`](../saddlellm/_CategoricalWorldModel.py#L314)<br><sub>`initial_posterior(self, embedding: torch.Tensor, sample: bool) -> Tuple[RSSMState, torch.Tensor]`</sub> | method | `CategoricalRSSMCell` 中实现`initial_posterior`的公开操作。 | `torch.zeros`, `self._regularized_logits`, `self.posterior`, `torch.cat`, `self._sample`, `RSSMState` |
| [`CategoricalRSSMCell.step`](../saddlellm/_CategoricalWorldModel.py#L329)<br><sub>`step(self, previous: RSSMState, action: torch.Tensor, embedding: Optional[torch.Tensor], sample: bool) -> Tuple[RSSMState, torch.Tensor, torch.Tensor]`</sub> | method | `CategoricalRSSMCell` 中实现`step`的公开操作。 | `self._core`, `self._regularized_logits`, `self.prior`, `self.posterior`, `torch.cat`, `self._sample`, `RSSMState` |
| [`CategoricalRSSMCell._core`](../saddlellm/_CategoricalWorldModel.py#L348)<br><sub>`_core(self, previous: RSSMState, action: torch.Tensor) -> torch.Tensor`</sub> | method | `CategoricalRSSMCell` 中实现`core`的内部辅助逻辑。 | `clamp_min`, `abs`, `action.detach`, `torch.cat`, `self.deterministic_input`, `self.stochastic_input`, `self.action_input`, `previous.deterministic.reshape`, `expand`, `shared.unsqueeze` |
| [`CategoricalRSSMCell._regularized_logits`](../saddlellm/_CategoricalWorldModel.py#L380)<br><sub>`_regularized_logits(self, flat_logits: torch.Tensor) -> torch.Tensor`</sub> | method | `CategoricalRSSMCell` 中实现`regularized_logits`的内部辅助逻辑。 | `flat_logits.reshape`, `torch.softmax`, `torch.log`, `probabilities.clamp_min` |
| [`CategoricalRSSMCell._sample`](../saddlellm/_CategoricalWorldModel.py#L392)<br><sub>`_sample(self, logits: torch.Tensor, sample: bool) -> torch.Tensor`</sub> | method | `CategoricalRSSMCell` 中采样`sample`的内部辅助逻辑。 | `logits.exp`, `sample`, `torch.distributions.Categorical`, `probabilities.argmax`, `to`, `F.one_hot`, `probabilities.detach`, `one_hot.flatten` |
| [`CategoricalWorldModel.__init__`](../saddlellm/_CategoricalWorldModel.py#L411)<br><sub>`__init__(self, config: CategoricalWorldModelConfig) -> None`</sub> | method | 初始化 `CategoricalWorldModel` 实例及其运行依赖。 | `__init__`, `super`, `ObservationEncoder`, `CategoricalRSSMCell`, `ObservationDecoder`, `_TwoHotSymlogHead`, `_mlp` |
| [`CategoricalWorldModel.forward`](../saddlellm/_CategoricalWorldModel.py#L426)<br><sub>`forward(self, observations: torch.Tensor, actions: torch.Tensor, sample_state: Optional[bool]=None) -> CategoricalWorldModelOutput`</sub> | method | 执行 `CategoricalWorldModel` 的前向计算。 | `self._validate_observations`, `self._prepare_actions`, `ValueError`, `bool`, `symlog`, `reshape`, `self.encoder`, `encoder_input.reshape`, `self.dynamics.initial_posterior`, `range` |
| [`CategoricalWorldModel.compute_loss`](../saddlellm/_CategoricalWorldModel.py#L492)<br><sub>`compute_loss(self, batch: Dict[str, torch.Tensor], sample_state: Optional[bool]=None) -> Dict[str, torch.Tensor]`</sub> | method | `CategoricalWorldModel` 中计算`compute_loss`的公开操作。 | `self`, `to`, `batch.get`, `torch.ones_like`, `symlog`, `tuple`, `range`, `mean`, `F.mse_loss`, `self._masked_mean` |
| [`CategoricalWorldModel.initial_state_from_observation`](../saddlellm/_CategoricalWorldModel.py#L553)<br><sub>`initial_state_from_observation(self, observation: torch.Tensor, sample_state: bool=False) -> RSSMState`</sub> | method | Encode one observation per batch into an initial posterior state. | `len`, `observation.unsqueeze`, `ValueError`, `tuple`, `observation.to`, `self._model_dtype`, `symlog`, `self.encoder`, `self.dynamics.initial_posterior`, `bool` |
| [`CategoricalWorldModel.imagine`](../saddlellm/_CategoricalWorldModel.py#L582)<br><sub>`imagine(self, initial_state: RSSMState, actions: torch.Tensor, deterministic: bool=True) -> WorldModelImagination`</sub> | method | `CategoricalWorldModel` 中实现`imagine`的公开操作。 | `self._prepare_actions`, `range`, `self.dynamics.step`, `self.decoder`, `symexp`, `self.reward_head`, `observations.append`, `rewards.append`, `continuation.append`, `torch.sigmoid` |
| [`CategoricalWorldModel.save_pretrained`](../saddlellm/_CategoricalWorldModel.py#L622)<br><sub>`save_pretrained(self, path: str) -> None`</sub> | method | 保存 `CategoricalWorldModel`，遵循预训练模型的目录契约。 | `os.makedirs`, `self.config.to_dict`, `open`, `os.path.join`, `json.dump`, `file.write`, `torch.save`, `self.state_dict` |
| [`CategoricalWorldModel.from_pretrained`](../saddlellm/_CategoricalWorldModel.py#L632)<br><sub>`from_pretrained(cls, path: str, map_location: Optional[Any]='cpu') -> 'CategoricalWorldModel'`</sub> | method | 从检查点加载 `CategoricalWorldModel`，遵循预训练模型的目录契约。 | `open`, `os.path.join`, `CategoricalWorldModelConfig.from_dict`, `json.load`, `cls`, `torch.load`, `model.load_state_dict` |
| [`CategoricalWorldModel.parameter_count`](../saddlellm/_CategoricalWorldModel.py#L649)<br><sub>`parameter_count(self, trainable_only: bool=False) -> int`</sub> | method | `CategoricalWorldModel` 中实现`parameter_count`的公开操作。 | `sum`, `parameter.numel`, `self.parameters` |
| [`CategoricalWorldModel._prepare_actions`](../saddlellm/_CategoricalWorldModel.py#L656)<br><sub>`_prepare_actions(self, actions: torch.Tensor) -> torch.Tensor`</sub> | method | `CategoricalWorldModel` 中准备`prepare_actions`的内部辅助逻辑。 | `actions.to`, `self._model_dtype`, `actions.squeeze`, `ValueError`, `actions.long`, `indices.numel`, `int`, `indices.min`, `indices.max`, `to` |
| [`CategoricalWorldModel._validate_observations`](../saddlellm/_CategoricalWorldModel.py#L681)<br><sub>`_validate_observations(self, observations: torch.Tensor) -> None`</sub> | method | `CategoricalWorldModel` 中校验`validate_observations`的内部辅助逻辑。 | `len`, `ValueError`, `tuple` |
| [`CategoricalWorldModel._model_dtype`](../saddlellm/_CategoricalWorldModel.py#L691)<br><sub>`_model_dtype(self) -> torch.dtype`</sub> | method | `CategoricalWorldModel` 中实现模型的内部辅助逻辑。 | `next`, `self.parameters` |
| [`CategoricalWorldModel._categorical_kl`](../saddlellm/_CategoricalWorldModel.py#L695)<br><sub>`_categorical_kl(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor`</sub> | method | `CategoricalWorldModel` 中实现`categorical_kl`的内部辅助逻辑。 | `sum`, `log_q.exp` |
| [`CategoricalWorldModel._masked_mean`](../saddlellm/_CategoricalWorldModel.py#L699)<br><sub>`_masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`</sub> | method | `CategoricalWorldModel` 中实现`masked_mean`的内部辅助逻辑。 | `sum`, `clamp_min`, `mask.sum` |

## `saddlellm/_WorldModel.py`

共 40 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`WorldModelConfig.__post_init__`](../saddlellm/_WorldModel.py#L69)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `WorldModelConfig` 创建后校验并规范化字段。 | `tuple`, `int`, `float`, `lower`, `str`, `self._validate` |
| [`WorldModelConfig.is_image`](../saddlellm/_WorldModel.py#L80)<br><sub>`is_image(self) -> bool`</sub> | method | `WorldModelConfig` 中实现图像的公开操作。 | `len` |
| [`WorldModelConfig.feature_size`](../saddlellm/_WorldModel.py#L84)<br><sub>`feature_size(self) -> int`</sub> | method | `WorldModelConfig` 中实现`feature_size`的公开操作。 | — |
| [`WorldModelConfig.to_dict`](../saddlellm/_WorldModel.py#L87)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldModelConfig` 转为可序列化字典。 | `asdict`, `list` |
| [`WorldModelConfig.from_dict`](../saddlellm/_WorldModel.py#L97)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'WorldModelConfig'`</sub> | method | 从字典解析并创建 `WorldModelConfig`。 | `isinstance`, `TypeError`, `fields`, `sorted`, `set`, `ValueError`, `join`, `cls`, `dict` |
| [`WorldModelConfig._validate`](../saddlellm/_WorldModel.py#L106)<br><sub>`_validate(self) -> None`</sub> | method | `WorldModelConfig` 中校验`validate`的内部辅助逻辑。 | `any`, `ValueError`, `getattr`, `len` |
| [`RSSMState.feature`](../saddlellm/_WorldModel.py#L176)<br><sub>`feature(self) -> torch.Tensor`</sub> | method | `RSSMState` 中实现`feature`的公开操作。 | `torch.cat` |
| [`RSSMState.detach`](../saddlellm/_WorldModel.py#L179)<br><sub>`detach(self) -> 'RSSMState'`</sub> | method | `RSSMState` 中实现`detach`的公开操作。 | `RSSMState`, `self.deterministic.detach`, `self.stochastic.detach`, `self.context.detach` |
| [`RSSMState.to`](../saddlellm/_WorldModel.py#L186)<br><sub>`to(self, *args: Any, **kwargs: Any) -> 'RSSMState'`</sub> | method | `RSSMState` 中实现`to`的公开操作。 | `RSSMState`, `self.deterministic.to`, `self.stochastic.to`, `self.context.to` |
| [`_activation`](../saddlellm/_WorldModel.py#L229)<br><sub>`_activation(name: str) -> nn.Module`</sub> | function | 模块级实现`activation`的内部辅助逻辑。 | `lower`, `str`, `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Tanh`, `nn.ELU`, `ValueError` |
| [`_MLP.__init__`](../saddlellm/_WorldModel.py#L245)<br><sub>`__init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, activation: str) -> None`</sub> | method | 初始化 `_MLP` 实例及其运行依赖。 | `__init__`, `super`, `range`, `max`, `layers.extend`, `nn.Linear`, `_activation`, `layers.append`, `nn.Sequential` |
| [`_MLP.forward`](../saddlellm/_WorldModel.py#L262)<br><sub>`forward(self, value: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `_MLP` 的前向计算。 | `self.network` |
| [`ObservationEncoder.__init__`](../saddlellm/_WorldModel.py#L269)<br><sub>`__init__(self, config: WorldModelConfig) -> None`</sub> | method | 初始化 `ObservationEncoder` 实例及其运行依赖。 | `__init__`, `super`, `int`, `reduce`, `_MLP`, `min`, `usable_channels.append`, `max`, `tuple`, `layers.extend` |
| [`ObservationEncoder.forward`](../saddlellm/_WorldModel.py#L311)<br><sub>`forward(self, observation: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `ObservationEncoder` 的前向计算。 | `self.encoder`, `observation.flatten`, `self.convolutions`, `self.projection`, `features.flatten` |
| [`ObservationDecoder.__init__`](../saddlellm/_WorldModel.py#L321)<br><sub>`__init__(self, config: WorldModelConfig, conv_channels: Sequence[int]) -> None`</sub> | method | 初始化 `ObservationDecoder` 实例及其运行依赖。 | `__init__`, `super`, `int`, `reduce`, `_MLP`, `tuple`, `len`, `max`, `math.ceil`, `nn.Linear` |
| [`ObservationDecoder.forward`](../saddlellm/_WorldModel.py#L379)<br><sub>`forward(self, feature: torch.Tensor) -> torch.Tensor`</sub> | method | 执行 `ObservationDecoder` 的前向计算。 | `self.decoder`, `value.reshape`, `self.direct_decoder`, `reshape`, `self.projection`, `self.deconvolutions`, `F.interpolate`, `torch.sigmoid` |
| [`RSSMCell.__init__`](../saddlellm/_WorldModel.py#L400)<br><sub>`__init__(self, config: WorldModelConfig) -> None`</sub> | method | 初始化 `RSSMCell` 实例及其运行依赖。 | `__init__`, `super`, `nn.Sequential`, `nn.Linear`, `_activation`, `nn.GRUCell`, `_MLP` |
| [`RSSMCell.zero_deterministic`](../saddlellm/_WorldModel.py#L423)<br><sub>`zero_deterministic(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor`</sub> | method | `RSSMCell` 中实现`zero_deterministic`的公开操作。 | `torch.zeros` |
| [`RSSMCell.initial_posterior`](../saddlellm/_WorldModel.py#L436)<br><sub>`initial_posterior(self, embedding: torch.Tensor, sample: bool=True) -> Tuple[RSSMState, torch.Tensor, torch.Tensor]`</sub> | method | `RSSMCell` 中实现`initial_posterior`的公开操作。 | `self.zero_deterministic`, `self._distribution`, `self.posterior`, `torch.cat`, `self._sample`, `RSSMState` |
| [`RSSMCell.step`](../saddlellm/_WorldModel.py#L448)<br><sub>`step(self, previous: RSSMState, action: torch.Tensor, embedding: Optional[torch.Tensor]=None, sample: bool=True) -> Tuple[RSSMState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`</sub> | method | `RSSMCell` 中实现`step`的公开操作。 | `self.recurrent_input`, `torch.cat`, `self.recurrent`, `self._distribution`, `self.prior`, `self.posterior`, `self._sample`, `RSSMState` |
| [`RSSMCell._distribution`](../saddlellm/_WorldModel.py#L470)<br><sub>`_distribution(self, parameters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`</sub> | method | `RSSMCell` 中实现`distribution`的内部辅助逻辑。 | `torch.chunk`, `F.softplus` |
| [`RSSMCell._sample`](../saddlellm/_WorldModel.py#L476)<br><sub>`_sample(mean: torch.Tensor, std: torch.Tensor, sample: bool) -> torch.Tensor`</sub> | method | `RSSMCell` 中采样`sample`的内部辅助逻辑。 | `torch.randn_like` |
| [`WorldModel.__init__`](../saddlellm/_WorldModel.py#L489)<br><sub>`__init__(self, config: WorldModelConfig) -> None`</sub> | method | 初始化 `WorldModel` 实例及其运行依赖。 | `__init__`, `super`, `ObservationEncoder`, `RSSMCell`, `ObservationDecoder`, `_MLP`, `replace`, `self._zero_output_layer` |
| [`WorldModel.forward`](../saddlellm/_WorldModel.py#L551)<br><sub>`forward(self, observations: torch.Tensor, actions: torch.Tensor, sample_state: Optional[bool]=None) -> WorldModelOutput`</sub> | method | Filter a batch of trajectories and predict every next transition. | `self._validate_observations`, `self._prepare_actions`, `ValueError`, `bool`, `observations.reshape`, `reshape`, `self.encoder`, `self.dynamics.initial_posterior`, `range`, `torch.cat` |
| [`WorldModel.compute_loss`](../saddlellm/_WorldModel.py#L655)<br><sub>`compute_loss(self, batch: Dict[str, torch.Tensor], sample_state: Optional[bool]=None) -> Dict[str, torch.Tensor]`</sub> | method | Compute reconstruction, reward, continuation and balanced KL losses. | `sorted`, `set`, `KeyError`, `join`, `self`, `batch.get`, `torch.ones_like`, `mask.to`, `ValueError`, `tuple` |
| [`WorldModel.initial_state_from_observation`](../saddlellm/_WorldModel.py#L796)<br><sub>`initial_state_from_observation(self, observation: torch.Tensor, sample_state: bool=False) -> RSSMState`</sub> | method | Encode one observation per batch into an initial posterior state. | `len`, `observation.unsqueeze`, `ValueError`, `tuple`, `self.encoder`, `observation.to`, `self._model_dtype`, `self.dynamics.initial_posterior`, `bool` |
| [`WorldModel.imagine`](../saddlellm/_WorldModel.py#L823)<br><sub>`imagine(self, initial_state: RSSMState, actions: torch.Tensor, deterministic: bool=True) -> WorldModelImagination`</sub> | method | Open-loop latent rollout from a posterior state and future actions. | `self._prepare_actions`, `ValueError`, `range`, `torch.cat`, `squeeze`, `self.collision_head`, `self._spatial_geometry_collision`, `collision_probability.append`, `torch.sigmoid`, `torch.where` |
| [`WorldModel._spatial_occupancy_logits`](../saddlellm/_WorldModel.py#L911)<br><sub>`_spatial_occupancy_logits(self, feature: torch.Tensor, context: Optional[torch.Tensor], action: torch.Tensor) -> torch.Tensor`</sub> | method | `WorldModel` 中实现`spatial_occupancy_logits`的内部辅助逻辑。 | `RuntimeError`, `self.occupancy_head`, `torch.tanh`, `to`, `torch.where`, `movement.abs`, `movement.sign`, `torch.zeros_like`, `self._spatial_geometry_collision`, `unsqueeze` |
| [`WorldModel._spatial_geometry_collision`](../saddlellm/_WorldModel.py#L966)<br><sub>`_spatial_geometry_collision(self, context: Optional[torch.Tensor], action: torch.Tensor) -> Optional[torch.Tensor]`</sub> | method | Detect an attempted collision from the agent-centred occupancy crop. | `torch.where`, `movement.abs`, `movement.sign`, `torch.zeros_like`, `to`, `torch.arange`, `blocked_at` |
| [`WorldModel._spatial_geometry_collision.blocked_at`](../saddlellm/_WorldModel.py#L991)<br><sub>`blocked_at(dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor`</sub> | nested function | `WorldModel` 中实现`blocked_at`的局部回调/辅助逻辑。 | `clamp` |
| [`WorldModel._zero_output_layer`](../saddlellm/_WorldModel.py#L1010)<br><sub>`_zero_output_layer(module: nn.Module) -> None`</sub> | method | `WorldModel` 中实现输出的内部辅助逻辑。 | `module.modules`, `isinstance`, `nn.init.zeros_` |
| [`WorldModel.save_pretrained`](../saddlellm/_WorldModel.py#L1020)<br><sub>`save_pretrained(self, path: str) -> None`</sub> | method | Save model weights plus a JSON network configuration. | `os.makedirs`, `self.config.to_dict`, `open`, `os.path.join`, `json.dump`, `file.write`, `torch.save`, `self.state_dict` |
| [`WorldModel.from_pretrained`](../saddlellm/_WorldModel.py#L1032)<br><sub>`from_pretrained(cls, path: str, map_location: Optional[Any]='cpu') -> 'WorldModel'`</sub> | method | Load a model written by :meth:`save_pretrained`. | `open`, `os.path.join`, `json.load`, `config_data.pop`, `WorldModelConfig.from_dict`, `cls`, `_safe_torch_load`, `model.state_dict`, `list`, `state.items` |
| [`WorldModel.parameter_count`](../saddlellm/_WorldModel.py#L1065)<br><sub>`parameter_count(self, trainable_only: bool=False) -> int`</sub> | method | `WorldModel` 中实现`parameter_count`的公开操作。 | `self.parameters`, `sum`, `parameter.numel` |
| [`WorldModel._validate_observations`](../saddlellm/_WorldModel.py#L1071)<br><sub>`_validate_observations(self, observations: torch.Tensor) -> None`</sub> | method | `WorldModel` 中校验`validate_observations`的内部辅助逻辑。 | `len`, `ValueError`, `tuple` |
| [`WorldModel._prepare_actions`](../saddlellm/_WorldModel.py#L1084)<br><sub>`_prepare_actions(self, actions: torch.Tensor) -> torch.Tensor`</sub> | method | `WorldModel` 中准备`prepare_actions`的内部辅助逻辑。 | `actions.to`, `self._model_dtype`, `actions.squeeze`, `ValueError`, `indices.numel`, `indices.min`, `indices.max`, `to`, `F.one_hot`, `actions.unsqueeze` |
| [`WorldModel._model_dtype`](../saddlellm/_WorldModel.py#L1112)<br><sub>`_model_dtype(self) -> torch.dtype`</sub> | method | `WorldModel` 中实现模型的内部辅助逻辑。 | `next`, `self.parameters` |
| [`WorldModel._normal_kl`](../saddlellm/_WorldModel.py#L1116)<br><sub>`_normal_kl(mean_q: torch.Tensor, std_q: torch.Tensor, mean_p: torch.Tensor, std_p: torch.Tensor) -> torch.Tensor`</sub> | method | `WorldModel` 中实现`normal_kl`的内部辅助逻辑。 | `pow`, `torch.log`, `kl.sum` |
| [`WorldModel._masked_mean`](../saddlellm/_WorldModel.py#L1128)<br><sub>`_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`</sub> | method | `WorldModel` 中实现`masked_mean`的内部辅助逻辑。 | `clamp_min`, `mask.sum`, `sum` |
| [`_safe_torch_load`](../saddlellm/_WorldModel.py#L1133)<br><sub>`_safe_torch_load(path: str, map_location: Optional[Any]) -> Any`</sub> | function | Load tensor-only files safely on new Torch and compatibly on old Torch. | `torch.load` |

## `saddlellm/_WorldModelTrainer.py`

共 31 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`WorldModelDataConfig.__post_init__`](../saddlellm/_WorldModelTrainer.py#L47)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `WorldModelDataConfig` 创建后校验并规范化字段。 | `ValueError` |
| [`WorldModelDataConfig.to_dict`](../saddlellm/_WorldModelTrainer.py#L57)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldModelDataConfig` 转为可序列化字典。 | `asdict` |
| [`WorldModelDataConfig.from_dict`](../saddlellm/_WorldModelTrainer.py#L61)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'WorldModelDataConfig'`</sub> | method | 从字典解析并创建 `WorldModelDataConfig`。 | `dict`, `values.pop`, `_reject_unknown_fields`, `cls` |
| [`WorldModelTrainingConfig.__post_init__`](../saddlellm/_WorldModelTrainer.py#L90)<br><sub>`__post_init__(self) -> None`</sub> | method | 在 `WorldModelTrainingConfig` 创建后校验并规范化字段。 | `ValueError`, `lower`, `str` |
| [`WorldModelTrainingConfig.to_dict`](../saddlellm/_WorldModelTrainer.py#L117)<br><sub>`to_dict(self) -> Dict[str, Any]`</sub> | method | 把 `WorldModelTrainingConfig` 转为可序列化字典。 | `asdict` |
| [`WorldModelTrainingConfig.from_dict`](../saddlellm/_WorldModelTrainer.py#L121)<br><sub>`from_dict(cls, data: Dict[str, Any]) -> 'WorldModelTrainingConfig'`</sub> | method | 从字典解析并创建 `WorldModelTrainingConfig`。 | `dict`, `values.pop`, `_reject_unknown_fields`, `cls` |
| [`WorldModelTrainer.__init__`](../saddlellm/_WorldModelTrainer.py#L147)<br><sub>`__init__(self, model: torch.nn.Module, train_dataset: WorldModelTrajectoryDataset, config: Optional[WorldModelTrainingConfig]=None, validation_dataset: Optional[WorldModelTrajectoryDataset]=None) -> None`</sub> | method | 初始化 `WorldModelTrainer` 实例及其运行依赖。 | `WorldModelTrainingConfig`, `_resolve_device`, `self.model.to`, `torch.optim.AdamW`, `self.model.parameters`, `_resolve_amp`, `_make_grad_scaler`, `float`, `self._seed_everything` |
| [`WorldModelTrainer.train`](../saddlellm/_WorldModelTrainer.py#L174)<br><sub>`train(self) -> Dict[str, Any]`</sub> | method | Run optimization and save the final model plus JSON metrics. | `os.makedirs`, `self._data_loader`, `math.ceil`, `len`, `min`, `torch.optim.lr_scheduler.LambdaLR`, `self._learning_rate_lambda`, `max`, `self._load_checkpoint`, `self.optimizer.zero_grad` |
| [`WorldModelTrainer.evaluate`](../saddlellm/_WorldModelTrainer.py#L337)<br><sub>`evaluate(self, loader: Optional[DataLoader]=None) -> Dict[str, float]`</sub> | method | Evaluate with posterior means for stable, reproducible metrics. | `ValueError`, `self._data_loader`, `self.model.eval`, `self._move_batch`, `self._autocast_context`, `self.model.compute_loss`, `float`, `cpu`, `detach`, `losses.items` |
| [`WorldModelTrainer._data_loader`](../saddlellm/_WorldModelTrainer.py#L366)<br><sub>`_data_loader(self, dataset: Optional[WorldModelTrajectoryDataset], shuffle: bool) -> DataLoader`</sub> | method | `WorldModelTrainer` 中实现数据的内部辅助逻辑。 | `ValueError`, `torch.Generator`, `generator.manual_seed`, `DataLoader` |
| [`WorldModelTrainer._move_batch`](../saddlellm/_WorldModelTrainer.py#L385)<br><sub>`_move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]`</sub> | method | `WorldModelTrainer` 中实现批次的内部辅助逻辑。 | `isinstance`, `value.to`, `batch.items` |
| [`WorldModelTrainer._autocast_context`](../saddlellm/_WorldModelTrainer.py#L393)<br><sub>`_autocast_context(self)`</sub> | method | `WorldModelTrainer` 中实现`autocast_context`的内部辅助逻辑。 | `nullcontext`, `torch.autocast` |
| [`WorldModelTrainer._learning_rate_lambda`](../saddlellm/_WorldModelTrainer.py#L398)<br><sub>`_learning_rate_lambda(self, total_steps: int)`</sub> | method | `WorldModelTrainer` 中实现`learning_rate_lambda`的内部辅助逻辑。 | `min` |
| [`WorldModelTrainer._learning_rate_lambda.schedule`](../saddlellm/_WorldModelTrainer.py#L401)<br><sub>`schedule(step: int) -> float`</sub> | nested function | `WorldModelTrainer` 中实现`schedule`的局部回调/辅助逻辑。 | `float`, `max`, `min`, `math.cos` |
| [`WorldModelTrainer._maybe_log`](../saddlellm/_WorldModelTrainer.py#L410)<br><sub>`_maybe_log(self, epoch: int, metrics: Dict[str, float]) -> None`</sub> | method | `WorldModelTrainer` 中记录`maybe_log`的内部辅助逻辑。 | `self.history.append`, `logger.info`, `metrics.get`, `float` |
| [`WorldModelTrainer._save_checkpoint`](../saddlellm/_WorldModelTrainer.py#L430)<br><sub>`_save_checkpoint(self, epoch: int) -> str`</sub> | method | `WorldModelTrainer` 中保存检查点的内部辅助逻辑。 | `RuntimeError`, `os.path.join`, `self.model.save_pretrained`, `self.optimizer.state_dict`, `self.scheduler.state_dict`, `self.scaler.state_dict`, `torch.save` |
| [`WorldModelTrainer._load_checkpoint`](../saddlellm/_WorldModelTrainer.py#L451)<br><sub>`_load_checkpoint(self, path: str) -> None`</sub> | method | `WorldModelTrainer` 中加载检查点的内部辅助逻辑。 | `RuntimeError`, `getattr`, `os.path.join`, `os.path.isfile`, `FileNotFoundError`, `_torch_load`, `self.model.load_state_dict`, `self.optimizer.load_state_dict`, `self.scheduler.load_state_dict`, `state.get` |
| [`WorldModelTrainer._save_json`](../saddlellm/_WorldModelTrainer.py#L474)<br><sub>`_save_json(self, filename: str, payload: Any) -> None`</sub> | method | `WorldModelTrainer` 中保存`save_json`的内部辅助逻辑。 | `os.path.join`, `open`, `json.dump`, `file.write` |
| [`WorldModelTrainer._seed_everything`](../saddlellm/_WorldModelTrainer.py#L481)<br><sub>`_seed_everything(seed: int) -> None`</sub> | method | `WorldModelTrainer` 中实现`seed_everything`的内部辅助逻辑。 | `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.is_available`, `torch.cuda.manual_seed_all` |
| [`prepare_world_model_training`](../saddlellm/_WorldModelTrainer.py#L489)<br><sub>`prepare_world_model_training(config: Union[str, os.PathLike, Dict[str, Any]]) -> PreparedWorldModelRun`</sub> | function | Load/validate a run config, infer dimensions, and construct datasets/model. | `_load_run_config`, `WorldModelDataConfig.from_dict`, `source_config.get`, `_resolve_input_path`, `WorldModelTrainingConfig.from_dict`, `load_world_model_trajectories`, `split_world_model_trajectories`, `dict`, `model_values.pop`, `normalize_world_model_backend` |
| [`train_world_model_from_config`](../saddlellm/_WorldModelTrainer.py#L602)<br><sub>`train_world_model_from_config(config: Union[str, os.PathLike, Dict[str, Any]], dry_run: bool=False) -> Dict[str, Any]`</sub> | function | High-level YAML/JSON/dict world-model training entry point. | `prepare_world_model_training`, `WorldModelTrainer`, `trainer.train`, `prepared.train_dataset.summary`, `prepared.validation_dataset.summary` |
| [`train_world_model`](../saddlellm/_WorldModelTrainer.py#L629)<br><sub>`train_world_model(config: Union[str, os.PathLike, Dict[str, Any]], dry_run: bool=False) -> Dict[str, Any]`</sub> | function | Alias for :func:`train_world_model_from_config`. | `train_world_model_from_config` |
| [`_load_run_config`](../saddlellm/_WorldModelTrainer.py#L638)<br><sub>`_load_run_config(config: Union[str, os.PathLike, Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[str]]`</sub> | function | 模块级加载配置的内部辅助逻辑。 | `isinstance`, `dict`, `os.fspath`, `os.path.isfile`, `FileNotFoundError`, `open`, `endswith`, `path.lower`, `yaml.safe_load`, `json.load` |
| [`_resolve_input_path`](../saddlellm/_WorldModelTrainer.py#L658)<br><sub>`_resolve_input_path(path: str, config_directory: Optional[str]) -> str`</sub> | function | 模块级解析输入、路径的内部辅助逻辑。 | `os.path.expandvars`, `os.path.expanduser`, `os.path.isfile`, `os.path.isabs`, `os.path.join` |
| [`_resolve_device`](../saddlellm/_WorldModelTrainer.py#L666)<br><sub>`_resolve_device(requested: str) -> torch.device`</sub> | function | 模块级解析设备的内部辅助逻辑。 | `lower`, `str`, `torch.cuda.is_available`, `torch.device`, `hasattr`, `torch.backends.mps.is_available`, `RuntimeError` |
| [`_resolve_amp`](../saddlellm/_WorldModelTrainer.py#L684)<br><sub>`_resolve_amp(requested: str, device: torch.device) -> Tuple[Optional[torch.dtype], bool]`</sub> | function | 模块级解析`resolve_amp`的内部辅助逻辑。 | `torch.cuda.is_bf16_supported`, `RuntimeError` |
| [`_make_grad_scaler`](../saddlellm/_WorldModelTrainer.py#L701)<br><sub>`_make_grad_scaler(enabled: bool)`</sub> | function | 模块级实现`make_grad_scaler`的内部辅助逻辑。 | `torch.amp.GradScaler`, `torch.cuda.amp.GradScaler` |
| [`_gradient_norm`](../saddlellm/_WorldModelTrainer.py#L708)<br><sub>`_gradient_norm(model: torch.nn.Module) -> float`</sub> | function | 模块级实现`gradient_norm`的内部辅助逻辑。 | `model.parameters`, `float`, `cpu`, `norm`, `parameter.grad.detach`, `math.sqrt` |
| [`_dtype_name`](../saddlellm/_WorldModelTrainer.py#L716)<br><sub>`_dtype_name(dtype: Optional[torch.dtype]) -> str`</sub> | function | 模块级实现`dtype_name`的内部辅助逻辑。 | — |
| [`_torch_load`](../saddlellm/_WorldModelTrainer.py#L724)<br><sub>`_torch_load(path: str, map_location: Any, weights_only: bool) -> Any`</sub> | function | 模块级加载`torch_load`的内部辅助逻辑。 | `torch.load` |
| [`_reject_unknown_fields`](../saddlellm/_WorldModelTrainer.py#L735)<br><sub>`_reject_unknown_fields(cls: Any, values: Dict[str, Any], label: str) -> None`</sub> | function | 模块级实现`reject_unknown_fields`的内部辅助逻辑。 | `fields`, `sorted`, `set`, `ValueError`, `join` |

## `saddlellm/__init__.py`

共 2 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`__getattr__`](../saddlellm/__init__.py#L544)<br><sub>`__getattr__(name)`</sub> | function | 按需解析未直接绑定的属性，主要用于延迟导入或兼容转发。 | `importlib.import_module`, `getattr`, `globals`, `AttributeError` |
| [`__dir__`](../saddlellm/__init__.py#L554)<br><sub>`__dir__()`</sub> | function | 返回该模块或兼容命名空间可发现的公开名称。 | `sorted`, `set`, `globals` |

## `saddlellm/cli.py`

共 31 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`main`](../saddlellm/cli.py#L9)<br><sub>`main(argv: Optional[list]=None) -> int`</sub> | function | 模块级实现`main`的公开操作。 | `argparse.ArgumentParser`, `parser.add_subparsers`, `sub.add_parser`, `init_p.add_argument`, `import_p.add_argument`, `plan_p.add_argument`, `train_p.add_argument`, `validate_p.add_argument`, `doctor_p.add_argument`, `smoke_p.add_argument` |
| [`_cmd_init`](../saddlellm/cli.py#L354)<br><sub>`_cmd_init(args) -> int`</sub> | function | 实现 CLI `init` 子命令，并返回进程退出码。 | `LLMTrainingFactory`, `FactoryConfig`, `factory.create_workspace`, `TrainingRecipe.template`, `os.path.join`, `recipe.save`, `print`, `json.dumps` |
| [`_cmd_import_data`](../saddlellm/cli.py#L381)<br><sub>`_cmd_import_data(args) -> int`</sub> | function | 实现 CLI `import-data` 子命令，并返回进程退出码。 | `item.strip`, `args.selected.split`, `LLMTrainingFactory`, `FactoryConfig`, `factory.import_dataset_manifest`, `print`, `json.dumps`, `len` |
| [`_cmd_plan`](../saddlellm/cli.py#L400)<br><sub>`_cmd_plan(args) -> int`</sub> | function | 实现 CLI `plan` 子命令，并返回进程退出码。 | `os.path.join`, `os.path.dirname`, `_load_config`, `SimpleFlowCompiler.is_simple_flow`, `SimpleFlowCompiler.compile`, `os.path.abspath`, `_save_config`, `TrainingRecipe.from_dict`, `recipe.save_compiled`, `get` |
| [`_cmd_train`](../saddlellm/cli.py#L438)<br><sub>`_cmd_train(args) -> int`</sub> | function | 实现 CLI `train` 子命令，并返回进程退出码。 | `_load_config`, `SimpleFlowCompiler.is_simple_flow`, `SimpleFlowCompiler.compile`, `os.path.dirname`, `os.path.abspath`, `os.path.join`, `_save_config`, `TrainingRecipe.from_dict`, `recipe.save_compiled`, `print` |
| [`_cmd_validate_config`](../saddlellm/cli.py#L496)<br><sub>`_cmd_validate_config(args) -> int`</sub> | function | 实现 CLI `validate-config` 子命令，并返回进程退出码。 | `os.path.exists`, `_emit_json`, `_load_config`, `SimpleFlowCompiler.is_simple_flow`, `SimpleFlowCompiler.compile`, `os.path.dirname`, `os.path.abspath`, `TrainingRecipe.from_dict`, `recipe.compile`, `to_dict` |
| [`_cmd_doctor`](../saddlellm/cli.py#L525)<br><sub>`_cmd_doctor(args) -> int`</sub> | function | 实现 CLI `doctor` 子命令，并返回进程退出码。 | `__import__`, `getattr`, `issues.append`, `sys.modules.get`, `bool`, `torch.cuda.is_available`, `torch.cuda.device_count`, `torch.cuda.get_device_name`, `torch.cuda.get_device_properties`, `torch.cuda.memory_allocated` |
| [`_cmd_smoke_test`](../saddlellm/cli.py#L624)<br><sub>`_cmd_smoke_test(args) -> int`</sub> | function | 实现 CLI `smoke-test` 子命令，并返回进程退出码。 | `run_smoke_tests`, `_emit_json`, `result.get` |
| [`_cmd_inspect_data`](../saddlellm/cli.py#L639)<br><sub>`_cmd_inspect_data(args) -> int`</sub> | function | 实现 CLI `inspect-data` 子命令，并返回进程退出码。 | `inspect_training_data`, `_emit_json`, `result.get` |
| [`_cmd_inspect_vla`](../saddlellm/cli.py#L647)<br><sub>`_cmd_inspect_vla(args) -> int`</sub> | function | 实现 CLI `inspect-vla` 子命令，并返回进程退出码。 | `VLAActionSpace`, `inspect_vla_data`, `_emit_json`, `result.get` |
| [`_cmd_train_world_model`](../saddlellm/cli.py#L669)<br><sub>`_cmd_train_world_model(args) -> int`</sub> | function | 实现 CLI `train-world-model` 子命令，并返回进程退出码。 | `train_world_model_from_config`, `_emit_json` |
| [`_cmd_build_media_cache`](../saddlellm/cli.py#L677)<br><sub>`_cmd_build_media_cache(args) -> int`</sub> | function | 实现 CLI `build-media-cache` 子命令，并返回进程退出码。 | `_load_config`, `isinstance`, `ValueError`, `os.path.dirname`, `os.path.abspath`, `payload.get`, `os.path.isabs`, `os.path.join`, `build_media_cache`, `MediaCacheBuildConfig` |
| [`_cmd_media_codecs`](../saddlellm/cli.py#L700)<br><sub>`_cmd_media_codecs(args) -> int`</sub> | function | 实现 CLI `media-codecs` 子命令，并返回进程退出码。 | `register_builtin_media_codecs`, `asdict`, `ModalityCodecRegistry.list_specs`, `_emit_json` |
| [`_cmd_world_model_backends`](../saddlellm/cli.py#L716)<br><sub>`_cmd_world_model_backends(args) -> int`</sub> | function | 实现 CLI `world-model-backends` 子命令，并返回进程退出码。 | `list_world_model_backends`, `_emit_json` |
| [`_cmd_infer_world_model`](../saddlellm/cli.py#L724)<br><sub>`_cmd_infer_world_model(args) -> int`</sub> | function | 实现 CLI `infer-world-model` 子命令，并返回进程退出码。 | `_load_config`, `run_world_model_inference`, `_emit_json` |
| [`_cmd_build_spatial_world_data`](../saddlellm/cli.py#L737)<br><sub>`_cmd_build_spatial_world_data(args) -> int`</sub> | function | 实现 CLI `build-spatial-world-data` 子命令，并返回进程退出码。 | `MapExtractionConfig`, `SpatialPlannerConfig`, `generate_spatial_sequence_dataset`, `SpatialSequenceConfig`, `generate_spatial_world_model_dataset`, `SpatialTrajectoryConfig`, `_emit_json` |
| [`_cmd_evaluate_spatial_world_model`](../saddlellm/cli.py#L800)<br><sub>`_cmd_evaluate_spatial_world_model(args) -> int`</sub> | function | 实现 CLI `evaluate-spatial-world-model` 子命令，并返回进程退出码。 | `evaluate_spatial_world_model`, `_emit_json` |
| [`_cmd_plan_spatial_route`](../saddlellm/cli.py#L816)<br><sub>`_cmd_plan_spatial_route(args) -> int`</sub> | function | 实现 CLI `plan-spatial-route` 子命令，并返回进程退出码。 | `QwenVLSpatialAnalyzer.from_pretrained`, `TopDownMapExtractor`, `MapExtractionConfig`, `GridPathPlanner`, `SpatialPlannerConfig`, `SpatialWorldModelCoordinator`, `coordinator.plan_image`, `_parse_spatial_point`, `os.path.splitext`, `os.path.basename` |
| [`_cmd_spatial_studio`](../saddlellm/cli.py#L893)<br><sub>`_cmd_spatial_studio(args) -> int`</sub> | function | 实现 CLI `spatial-studio` 子命令，并返回进程退出码。 | `ImportError`, `SpatialStudioSettings`, `create_spatial_studio_app`, `uvicorn.run` |
| [`_world_agent_settings`](../saddlellm/cli.py#L916)<br><sub>`_world_agent_settings(args)`</sub> | function | 模块级实现世界的内部辅助逻辑。 | `getattr`, `WorldAgentSettings.from_file`, `WorldAgentSettings`, `setattr` |
| [`_cmd_world_agent_run`](../saddlellm/cli.py#L936)<br><sub>`_cmd_world_agent_run(args) -> int`</sub> | function | 实现 CLI `world-agent-run` 子命令，并返回进程退出码。 | `_world_agent_settings`, `WorldAgentRuntime`, `runtime.run_image`, `_parse_spatial_point`, `_emit_json` |
| [`_cmd_world_agent_api`](../saddlellm/cli.py#L955)<br><sub>`_cmd_world_agent_api(args) -> int`</sub> | function | 实现 CLI `world-agent-api` 子命令，并返回进程退出码。 | `ImportError`, `_world_agent_settings`, `create_world_agent_app`, `uvicorn.run` |
| [`_cmd_preflight`](../saddlellm/cli.py#L968)<br><sub>`_cmd_preflight(args) -> int`</sub> | function | 实现 CLI `preflight` 子命令，并返回进程退出码。 | `LLMTrainingFactory`, `FactoryConfig`, `factory.create_preflight_plan`, `_emit_json`, `result.get` |
| [`_cmd_create_recipe`](../saddlellm/cli.py#L1001)<br><sub>`_cmd_create_recipe(args) -> int`</sub> | function | 实现 CLI `create-recipe` 子命令，并返回进程退出码。 | `TrainingRecipe.template`, `os.path.dirname`, `os.path.abspath`, `bool`, `recipe.save`, `print`, `json.dumps`, `recipe.stages` |
| [`_cmd_report`](../saddlellm/cli.py#L1019)<br><sub>`_cmd_report(args) -> int`</sub> | function | 实现 CLI `report` 子命令，并返回进程退出码。 | `report`, `LLMTrainingFactory`, `FactoryConfig`, `print`, `json.dumps` |
| [`_cmd_export_model`](../saddlellm/cli.py#L1028)<br><sub>`_cmd_export_model(args) -> int`</sub> | function | 实现 CLI `export-model` 子命令，并返回进程退出码。 | `_load_config`, `isinstance`, `gate.get`, `_emit_json`, `to_dict`, `ModelExporter.export`, `ModelExportRequest`, `type`, `str` |
| [`_cmd_serve_model`](../saddlellm/cli.py#L1074)<br><sub>`_cmd_serve_model(args) -> int`</sub> | function | 实现 CLI `serve-model` 子命令，并返回进程退出码。 | `ImportError`, `os.environ.get`, `ValueError`, `InferenceServerSettings`, `create_inference_app`, `uvicorn.run` |
| [`_emit_json`](../saddlellm/cli.py#L1111)<br><sub>`_emit_json(payload: dict, output: Optional[str]=None) -> None`</sub> | function | 模块级实现`emit_json`的内部辅助逻辑。 | `json.dumps`, `os.makedirs`, `os.path.dirname`, `open`, `f.write`, `print` |
| [`_parse_spatial_point`](../saddlellm/cli.py#L1121)<br><sub>`_parse_spatial_point(value: str)`</sub> | function | 模块级解析`parse_spatial_point`的内部辅助逻辑。 | `strip`, `str`, `part.strip`, `text.split`, `len`, `ValueError`, `float` |
| [`_load_config`](../saddlellm/cli.py#L1134)<br><sub>`_load_config(path: str) -> dict`</sub> | function | 模块级加载配置的内部辅助逻辑。 | `open`, `endswith`, `path.lower`, `yaml.safe_load`, `json.load` |
| [`_save_config`](../saddlellm/cli.py#L1142)<br><sub>`_save_config(path: str, payload: dict) -> str`</sub> | function | 模块级保存配置的内部辅助逻辑。 | `os.makedirs`, `os.path.dirname`, `open`, `endswith`, `path.lower`, `yaml.safe_dump`, `json.dump`, `f.write` |

## `saddlellm/distill.py`

共 11 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`DistillationTrainer.__init__`](../saddlellm/distill.py#L106)<br><sub>`__init__(self, teacher_model=None, temperature=2.0, alpha=0.5, *args, **kwargs)`</sub> | method | 初始化 `DistillationTrainer` 实例及其运行依赖。 | `__init__`, `super`, `self.teacher.eval` |
| [`DistillationTrainer.compute_loss`](../saddlellm/distill.py#L113)<br><sub>`compute_loss(self, model, inputs, return_outputs=False, **kwargs)`</sub> | method | 计算蒸馏损失 (改进版) | `inspect.signature`, `list`, `signature.parameters.keys`, `inputs.items`, `logger.warning`, `model`, `torch.no_grad`, `self.teacher`, `F.kl_div`, `F.log_softmax` |
| [`ModelDistiller.__init__`](../saddlellm/distill.py#L165)<br><sub>`__init__(self, args: DistillationArguments)`</sub> | method | 初始化 `ModelDistiller` 实例及其运行依赖。 | `torch.device`, `torch.cuda.is_available`, `logger.info`, `AutoModelForCausalLM.from_pretrained`, `self.teacher.to`, `self.student.to`, `AutoTokenizer.from_pretrained` |
| [`ModelDistiller.format_prompt`](../saddlellm/distill.py#L184)<br><sub>`format_prompt(self, example)`</sub> | method | 格式化提示 | `isinstance`, `example.get`, `self.args.prompt_template.format` |
| [`ModelDistiller.generate_sample_data`](../saddlellm/distill.py#L204)<br><sub>`generate_sample_data(self, num_samples=1000)`</sub> | method | 生成模拟数据（当没有真实数据时使用） | `logger.info`, `range`, `random.choice`, `base_sample.copy`, `len`, `replace`, `samples.append`, `Dataset.from_list` |
| [`ModelDistiller.load_custom_dataset`](../saddlellm/distill.py#L242)<br><sub>`load_custom_dataset(self)`</sub> | method | 加载自定义数据集 | `logger.info`, `os.path.exists`, `logger.warning`, `self.generate_sample_data`, `open`, `json.loads`, `line.strip`, `isinstance`, `str`, `item.get` |
| [`ModelDistiller.prepare_dataset`](../saddlellm/distill.py#L286)<br><sub>`prepare_dataset(self)`</sub> | method | 准备数据集 | `self.load_custom_dataset`, `logger.info`, `load_dataset`, `dataset.map`, `logger.error`, `isinstance`, `tokenized_datasets.train_test_split`, `train_test_split` |
| [`ModelDistiller.prepare_dataset.tokenize_function`](../saddlellm/distill.py#L299)<br><sub>`tokenize_function(examples)`</sub> | nested function | `ModelDistiller` 中分词`tokenize_function`的局部回调/辅助逻辑。 | `isinstance`, `str`, `examples.get`, `len`, `range`, `self.format_prompt`, `prompts.append`, `labels.append`, `self.tokenizer` |
| [`ModelDistiller.distill`](../saddlellm/distill.py#L370)<br><sub>`distill(self)`</sub> | method | 执行蒸馏过程 | `self.prepare_dataset`, `logger.error`, `TrainingArguments`, `torch.cuda.is_available`, `DistillationTrainer`, `tokenized_datasets.get`, `logger.info`, `trainer.train`, `trainer.save_model`, `os.path.join` |
| [`parse_args`](../saddlellm/distill.py#L426)<br><sub>`parse_args()`</sub> | function | 解析命令行参数 | `argparse.ArgumentParser`, `parser.add_argument`, `parser.set_defaults`, `parser.parse_args` |
| [`test_distillation`](../saddlellm/distill.py#L477)<br><sub>`test_distillation()`</sub> | function | 测试蒸馏过程 | `DistillationArguments`, `ModelDistiller`, `distiller.distill`, `logger.info`, `logger.error`, `str` |

## `saddlellm/easy.py`

共 12 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`create`](../saddlellm/easy.py#L33)<br><sub>`create(size: str='300m', architecture: str='llama', vocab_size: int=None) -> 'PreTrainedModel'`</sub> | function | 创建模型。只需指定大小。 | `size_map.get`, `size.lower`, `ModelRegistry.create_model`, `print`, `spec.human_params`, `size.endswith`, `float`, `ValueError`, `min`, `MODEL_SPECS.values` |
| [`train`](../saddlellm/easy.py#L99)<br><sub>`train(model, tokenizer=None, data=None, steps: int=10000, learning_rate: float=None, output_dir: str='./my_model', use_ema: bool=True, data_lang: str='zh', **kwargs)`</sub> | function | 训练模型。自动处理数据、优化器、EMA。 | `AutoTokenizer.from_pretrained`, `print`, `DataCatalog.small_model_pack`, `isinstance`, `os.path.exists`, `os.path.splitext`, `_load_data`, `DataCatalog.fetch`, `sum`, `p.numel` |
| [`train_tokenizer`](../saddlellm/easy.py#L208)<br><sub>`train_tokenizer(data_path: str, vocab_size: int=32000, output_dir: str='./my_tokenizer')`</sub> | function | 训练 BPE 分词器。 | `TokenizerTrainer`, `trainer.fit`, `trainer.save`, `trainer.get_hf_tokenizer`, `print` |
| [`distill`](../saddlellm/easy.py#L227)<br><sub>`distill(model, tokenizer=None, teacher: str='gpt-4o', prompts=None, api_key: str=None, examples: int=500, output_dir: str='./my_distilled_model')`</sub> | function | 从强模型蒸馏能力。 | `AutoTokenizer.from_pretrained`, `TeacherInterface.from_openai`, `TeacherInterface.from_anthropic`, `TeacherInterface.from_deepseek`, `os.path.exists`, `AutoModelForCausalLM.from_pretrained`, `AT.from_pretrained`, `TeacherInterface.from_local`, `range`, `AutoDistiller` |
| [`chat`](../saddlellm/easy.py#L303)<br><sub>`chat(model, prompt: str, tokenizer=None, max_new_tokens: int=512, temperature: float=0.7, system_prompt: str=None, use_cot: bool=False, use_self_consistency: bool=False) -> str`</sub> | function | 与模型对话。 | `AutoTokenizer.from_pretrained`, `SelfConsistency`, `sc.solve`, `SafeGenerate`, `sg.generate` |
| [`improve`](../saddlellm/easy.py#L357)<br><sub>`improve(model, tokenizer=None, data=None, steps: int=5000, output_dir: str='./my_improved_model')`</sub> | function | 一键应用所有效果提升技术。 | `AutoTokenizer.from_pretrained`, `print`, `EvolInstruct`, `evolver.evolve_batch`, `isinstance`, `len`, `train`, `SelfConsistency`, `sc.solve` |
| [`evaluate`](../saddlellm/easy.py#L413)<br><sub>`evaluate(model, tokenizer=None, tasks=None)`</sub> | function | 快速评估模型。 | `AutoTokenizer.from_pretrained`, `BenchmarkRunner`, `runner.run`, `print` |
| [`list_models`](../saddlellm/easy.py#L439)<br><sub>`list_models()`</sub> | function | 列出所有可用的模型架构。 | `print`, `ModelRegistry.compare_specs` |
| [`check`](../saddlellm/easy.py#L453)<br><sub>`check()`</sub> | function | 运行环境自检。 | `check_environment` |
| [`auto`](../saddlellm/easy.py#L462)<br><sub>`auto(model_size: str='300m', lang: str='zh', output_dir: str='./my_model', use_api: bool=False)`</sub> | function | 一键训练最佳小模型。 | `auto_train_best_small_model` |
| [`auto_config`](../saddlellm/easy.py#L473)<br><sub>`auto_config(model=None, model_size: str=None)`</sub> | function | 自动检测 GPU 并推荐最佳训练配置。 | `torch.cuda.is_available`, `torch.cuda.device_count`, `torch.cuda.get_device_name`, `torch.cuda.get_device_properties`, `get`, `next`, `print`, `sum`, `p.numel`, `model.parameters` |
| [`_load_data`](../saddlellm/easy.py#L582)<br><sub>`_load_data(data_path: str, tokenizer)`</sub> | function | 加载和预处理数据。 | `PipelineConfig`, `DataPipeline`, `filter_quality`, `deduplicate`, `clean`, `pipeline.collect`, `pipeline.tokenize_and_pack`, `pipeline.to_iterable_dataset` |

## `saddlellm/load_model.py`

共 3 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`unsloth_load_lora_model`](../saddlellm/load_model.py#L3)<br><sub>`unsloth_load_lora_model(MODEL_PATH)`</sub> | function | 模块级加载模型的公开操作。 | `print`, `torch.device`, `torch.cuda.is_available`, `FastLanguageModel.from_pretrained`, `model.to`, `FastLanguageModel.for_inference`, `to`, `tokenizer`, `alpaca_prompt.format`, `TextStreamer` |
| [`transfomer_load_lora`](../saddlellm/load_model.py#L72)<br><sub>`transfomer_load_lora(MODEL_PATH)`</sub> | function | 模块级加载`transfomer_load_lora`的公开操作。 | `print`, `BitsAndBytesConfig`, `AutoPeftModelForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained`, `alpaca_prompt.format`, `to`, `tokenizer`, `model.generate`, `tokenizer.decode` |
| [`src_mode_transformer_load`](../saddlellm/load_model.py#L113)<br><sub>`src_mode_transformer_load(MODEL_PATH='E:\\reactflow_test\\backend\\model\\lora_model_step_120')`</sub> | function | 模块级加载`src_mode_transformer_load`的公开操作。 | `print`, `torch.cuda.is_available`, `os.path.abspath`, `AutoModelForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained`, `transformers.pipeline`, `tokenizer.apply_chat_template`, `pipeline`, `len` |

## `saddlellm/monitordashbord.py`

共 18 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MonitorServer.__init__`](../saddlellm/monitordashbord.py#L63)<br><sub>`__init__(self)`</sub> | method | 初始化 `MonitorServer` 实例及其运行依赖。 | — |
| [`MonitorServer.collect_system_metrics`](../saddlellm/monitordashbord.py#L71)<br><sub>`async collect_system_metrics(self)`</sub> | method | 收集系统指标 | `psutil.cpu_percent`, `self._get_cpu_temp`, `psutil.cpu_count`, `psutil.virtual_memory`, `pynvml.nvmlDeviceGetHandleByIndex`, `pynvml.nvmlDeviceGetUtilizationRates`, `pynvml.nvmlDeviceGetTemperature`, `set`, `GPU_UTIL.labels`, `GPU_MEM.labels` |
| [`MonitorServer._get_cpu_temp`](../saddlellm/monitordashbord.py#L119)<br><sub>`_get_cpu_temp(self) -> Optional[float]`</sub> | method | 获取CPU温度(Linux系统) | `open`, `float`, `f.read` |
| [`MonitorServer.check_anomalies`](../saddlellm/monitordashbord.py#L128)<br><sub>`async check_anomalies(self, status: SystemStatus)`</sub> | method | 检查系统异常 | `alerts.append`, `AlertData`, `time.time`, `self.broadcast_alert`, `self.alerts.append` |
| [`MonitorServer.broadcast_system_status`](../saddlellm/monitordashbord.py#L172)<br><sub>`async broadcast_system_status(self, status: SystemStatus)`</sub> | method | 广播系统状态 | `status.dict`, `self._broadcast` |
| [`MonitorServer.broadcast_metrics`](../saddlellm/monitordashbord.py#L180)<br><sub>`async broadcast_metrics(self, mode: str, metrics: List[MetricData])`</sub> | method | 广播指标数据 | `m.dict`, `self._broadcast`, `extend`, `len` |
| [`MonitorServer.broadcast_alert`](../saddlellm/monitordashbord.py#L196)<br><sub>`async broadcast_alert(self, alert: AlertData)`</sub> | method | 广播告警 | `alert.dict`, `self._broadcast` |
| [`MonitorServer._broadcast`](../saddlellm/monitordashbord.py#L204)<br><sub>`async _broadcast(self, message: Dict)`</sub> | method | 广播消息给所有客户端 | `self.active_connections.values`, `connection.send_text`, `json.dumps` |
| [`MonitorServer.connect`](../saddlellm/monitordashbord.py#L212)<br><sub>`async connect(self, websocket: WebSocket, client_id: str)`</sub> | method | 处理新客户端连接 | `websocket.accept`, `websocket.send_text`, `json.dumps` |
| [`MonitorServer.disconnect`](../saddlellm/monitordashbord.py#L227)<br><sub>`disconnect(self, client_id: str)`</sub> | method | 处理客户端断开连接 | `self.active_connections.pop` |
| [`startup_event`](../saddlellm/monitordashbord.py#L236)<br><sub>`async startup_event()`</sub> | function | 模块级实现`startup_event`的公开操作。 | `asyncio.create_task`, `monitor_server.collect_system_metrics` |
| [`shutdown_event`](../saddlellm/monitordashbord.py#L241)<br><sub>`async shutdown_event()`</sub> | function | 模块级实现`shutdown_event`的公开操作。 | `pynvml.nvmlShutdown` |
| [`websocket_endpoint`](../saddlellm/monitordashbord.py#L247)<br><sub>`async websocket_endpoint(websocket: WebSocket, client_id: str)`</sub> | function | 模块级实现`websocket_endpoint`的公开操作。 | `monitor_server.connect`, `websocket.receive_text`, `monitor_server.disconnect` |
| [`get_metrics`](../saddlellm/monitordashbord.py#L257)<br><sub>`async get_metrics()`</sub> | function | 模块级读取指标的公开操作。 | `CPU_UTIL._value.get`, `MEM_USAGE._value.get`, `TRAINING_LOSS._value.get`, `THROUGHPUT._value.get` |
| [`get_alerts`](../saddlellm/monitordashbord.py#L267)<br><sub>`async get_alerts(limit: int=50)`</sub> | function | 模块级读取`get_alerts`的公开操作。 | `a.dict`, `len` |
| [`push_metric`](../saddlellm/monitordashbord.py#L273)<br><sub>`async push_metric(metric: MetricData)`</sub> | function | 模块级实现`push_metric`的公开操作。 | `TRAINING_LOSS.set`, `THROUGHPUT.inc` |
| [`get_status`](../saddlellm/monitordashbord.py#L282)<br><sub>`async get_status()`</sub> | function | 模块级读取`get_status`的公开操作。 | `psutil.virtual_memory`, `psutil.cpu_percent`, `monitor_server._get_cpu_temp`, `psutil.cpu_count`, `pynvml.nvmlDeviceGetHandleByIndex`, `pynvml.nvmlDeviceGetUtilizationRates`, `pynvml.nvmlDeviceGetTemperature` |
| [`dashboard`](../saddlellm/monitordashbord.py#L302)<br><sub>`async dashboard()`</sub> | function | 模块级实现`dashboard`的公开操作。 | `HTMLResponse` |

## `saddlellm/pretrain.py`

共 5 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`load_pretrained_model`](../saddlellm/pretrain.py#L26)<br><sub>`load_pretrained_model(model_path)`</sub> | function | 加载预训练模型 | `AutoModelForCausalLM.from_pretrained` |
| [`create_model_from_scratch`](../saddlellm/pretrain.py#L35)<br><sub>`create_model_from_scratch(model_type)`</sub> | function | 从头创建模型 | `LlamaConfig`, `LlamaForCausalLM`, `GPTNeoXConfig`, `GPTNeoXForCausalLM`, `ValueError` |
| [`load_data`](../saddlellm/pretrain.py#L58)<br><sub>`load_data(data_path, data_format)`</sub> | function | 加载训练数据 | `load_dataset`, `ValueError` |
| [`train`](../saddlellm/pretrain.py#L68)<br><sub>`train(args)`</sub> | function | 模块级训练`train`的公开操作。 | `torch.cuda.is_available`, `logger.info`, `load_pretrained_model`, `create_model_from_scratch`, `model.to`, `AutoTokenizer.from_pretrained`, `load_data`, `dataset.map`, `TrainingArguments`, `DataCollatorForLanguageModeling` |
| [`train.tokenize_function`](../saddlellm/pretrain.py#L95)<br><sub>`tokenize_function(examples)`</sub> | nested function | 模块级分词`tokenize_function`的局部回调/辅助逻辑。 | `tokenizer` |

## `saddlellm/prune.py`

共 21 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`ModelPruner.get_prunable_layers`](../saddlellm/prune.py#L38)<br><sub>`get_prunable_layers(model: nn.Module, target_modules: List[str]) -> List[Tuple[str, nn.Module]]`</sub> | method | 获取可剪枝的层，增加设备兼容性检查 | `model.named_modules`, `any`, `isinstance`, `prunable_layers.append`, `logger.info`, `len` |
| [`ModelPruner.apply_pruning`](../saddlellm/prune.py#L49)<br><sub>`apply_pruning(model: nn.Module, pruning_method: str, pruning_config: Dict, target_modules: List[str]=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']) -> nn.Module`</sub> | method | 应用静态剪枝，优化设备内存管理 | `ModelPruner.get_prunable_layers`, `logger.warning`, `prune.l1_unstructured`, `prune.random_unstructured`, `pruning_config.get`, `prune.ln_structured`, `prune.global_unstructured` |
| [`ModelPruner.remove_pruning`](../saddlellm/prune.py#L85)<br><sub>`remove_pruning(model: nn.Module) -> nn.Module`</sub> | method | 移除剪枝掩码，永久应用剪枝，增加内存清理 | `model.named_modules`, `prune.is_pruned`, `prune.remove`, `torch.cuda.empty_cache` |
| [`ModelPruner.get_sparsity`](../saddlellm/prune.py#L94)<br><sub>`get_sparsity(model: nn.Module) -> float`</sub> | method | 计算模型稀疏度，优化计算效率 | `model.named_modules`, `isinstance`, `hasattr`, `module.weight.cpu`, `item`, `torch.sum`, `weight_cpu.numel` |
| [`PruningTrainer.__init__`](../saddlellm/prune.py#L111)<br><sub>`__init__(self, pruning_method: str, pruning_config: Dict, **kwargs)`</sub> | method | 初始化 `PruningTrainer` 实例及其运行依赖。 | `__init__`, `super`, `pruning_method.lower`, `float`, `self._create_prune_schedule`, `self._init_pruning` |
| [`PruningTrainer.compute_loss`](../saddlellm/prune.py#L124)<br><sub>`compute_loss(self, model, inputs, return_outputs=False, **kwargs)`</sub> | method | 计算损失并更新剪枝 | `inputs.items`, `isinstance`, `to`, `model`, `self._apply_pruning`, `deepcopy`, `model.state_dict`, `cpu` |
| [`PruningTrainer._create_prune_schedule`](../saddlellm/prune.py#L166)<br><sub>`_create_prune_schedule(self) -> List[float]`</sub> | method | 创建剪枝比例调度，增加边界检查 | `max`, `np.linspace`, `np.cos`, `np.arange` |
| [`PruningTrainer._init_pruning`](../saddlellm/prune.py#L178)<br><sub>`_init_pruning(self)`</sub> | method | 初始化剪枝，优化设备管理 | `self.pruning_config.get`, `ModelPruner.get_prunable_layers`, `hasattr`, `module.register_buffer`, `torch.zeros_like`, `logger.info`, `len` |
| [`PruningTrainer._get_current_prune_ratio`](../saddlellm/prune.py#L194)<br><sub>`_get_current_prune_ratio(self) -> float`</sub> | method | 获取当前剪枝比例，增加边界处理 | `max`, `min` |
| [`PruningTrainer._apply_pruning`](../saddlellm/prune.py#L200)<br><sub>`_apply_pruning(self)`</sub> | method | 应用剪枝策略，优化内存使用 | `self._get_current_prune_ratio`, `self._magnitude_pruning`, `self._movement_pruning`, `self._structured_pruning`, `torch.cuda.empty_cache` |
| [`PruningTrainer._magnitude_pruning`](../saddlellm/prune.py#L214)<br><sub>`_magnitude_pruning(self, ratio: float)`</sub> | method | 幅度剪枝，优化循环效率 | `prune.l1_unstructured` |
| [`PruningTrainer._movement_pruning`](../saddlellm/prune.py#L219)<br><sub>`_movement_pruning(self, ratio: float)`</sub> | method | 动态运动剪枝，优化梯度处理 | `torch.abs`, `torch.quantile`, `module.importance_momentum.flatten`, `hasattr`, `mask.to`, `prune.custom_from_mask` |
| [`PruningTrainer._structured_pruning`](../saddlellm/prune.py#L241)<br><sub>`_structured_pruning(self, ratio: float)`</sub> | method | 结构化剪枝，增加维度验证 | `self.pruning_config.get`, `logger.warning`, `prune.ln_structured` |
| [`PruningTrainer.evaluate_model`](../saddlellm/prune.py#L290)<br><sub>`evaluate_model(self, eval_dataset=None)`</sub> | method | 评估剪枝模型性能，优化评估流程 | `ModelPruner.get_sparsity`, `evaluate`, `super` |
| [`load_model_and_tokenizer`](../saddlellm/prune.py#L298)<br><sub>`load_model_and_tokenizer(model_path: str, device: str='auto')`</sub> | function | 加载模型和分词器，优化设备分配 | `logger.info`, `torch.cuda.is_available`, `logger.warning`, `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained` |
| [`prepare_dataset`](../saddlellm/prune.py#L325)<br><sub>`prepare_dataset(tokenizer, data_path: str, seq_length: int=512)`</sub> | function | 准备训练数据集，优化数据处理流程 | `logger.info`, `data_path.endswith`, `load_dataset`, `os.path.isdir`, `dataset.map`, `logger.warning`, `str`, `all_texts.extend`, `list`, `values` |
| [`prepare_dataset.tokenize_function`](../saddlellm/prune.py#L342)<br><sub>`tokenize_function(examples)`</sub> | nested function | 模块级分词`tokenize_function`的局部回调/辅助逻辑。 | `list`, `examples.values`, `tokenizer` |
| [`train`](../saddlellm/prune.py#L397)<br><sub>`train(args)`</sub> | function | 执行剪枝训练，优化内存管理和错误处理 | `set_seed`, `load_model_and_tokenizer`, `logger.error`, `str`, `prepare_dataset`, `train_test_split`, `TrainingArguments`, `os.path.join`, `DataCollatorForLanguageModeling`, `PruningTrainer` |
| [`evaluate_pruned_model`](../saddlellm/prune.py#L536)<br><sub>`evaluate_pruned_model(model_path: str, eval_data: str, device: str='auto')`</sub> | function | 评估剪枝后的模型，优化设备管理和错误处理 | `logger.info`, `load_model_and_tokenizer`, `logger.error`, `str`, `prepare_dataset`, `ModelPruner.get_sparsity`, `TrainingArguments`, `Trainer`, `DataCollatorForLanguageModeling`, `trainer.evaluate` |
| [`parse_args`](../saddlellm/prune.py#L589)<br><sub>`parse_args()`</sub> | function | 解析命令行参数，增加参数验证 | `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `logger.error`, `exit`, `os.path.exists` |
| [`main`](../saddlellm/prune.py#L679)<br><sub>`main()`</sub> | function | 主函数，优化错误处理流程 | `parse_args`, `os.makedirs`, `logger.error`, `str`, `train`, `evaluate_pruned_model`, `os.path.join` |

## `saddlellm/quantize.py`

共 5 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`load_model`](../saddlellm/quantize.py#L28)<br><sub>`load_model(model_path, device)`</sub> | function | 加载原始模型 | `logger.info`, `AutoModelForCausalLM.from_pretrained`, `model.eval`, `logger.error`, `str` |
| [`prepare_calib_data`](../saddlellm/quantize.py#L45)<br><sub>`prepare_calib_data(tokenizer, data_path, num_samples=128, seq_len=128)`</sub> | function | 准备校准数据，返回文本列表 | `logger.info`, `open`, `f.readlines`, `tqdm`, `range`, `min`, `len`, `json.loads`, `data.get`, `text.strip` |
| [`quantize_model`](../saddlellm/quantize.py#L71)<br><sub>`quantize_model(args, model, tokenizer)`</sub> | function | 执行模型量化，改进数据集格式 | `args.method.lower`, `os.path.abspath`, `os.makedirs`, `logger.info`, `quant_method.upper`, `BitsAndBytesConfig`, `torch.cuda.empty_cache`, `AutoModelForCausalLM.from_pretrained`, `model.save_pretrained`, `tokenizer.save_pretrained` |
| [`test_model`](../saddlellm/quantize.py#L199)<br><sub>`test_model(model_path, tokenizer, prompt=None)`</sub> | function | 测试量化模型 | `logger.info`, `os.path.exists`, `os.path.join`, `AutoModelForCausalLM.from_pretrained`, `BitsAndBytesConfig.from_pretrained`, `to`, `tokenizer`, `torch.no_grad`, `model.generate`, `tokenizer.decode` |
| [`main`](../saddlellm/quantize.py#L260)<br><sub>`main()`</sub> | function | 模块级实现`main`的公开操作。 | `argparse.ArgumentParser`, `parser.add_argument`, `parser.parse_args`, `os.path.exists`, `logger.error`, `torch.cuda.is_available`, `logger.info`, `load_model`, `AutoTokenizer.from_pretrained`, `test_model` |

## `spatial-studio/src/App.tsx`

共 14 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`App`](../spatial-studio/src/App.tsx#L39)<br><sub>`App()`</sub> | function/component | 模块级实现`App`的公开操作。 | — |
| [`restore`](../spatial-studio/src/App.tsx#L93)<br><sub>`restore()`</sub> | local function | `App` 中恢复`restore`的局部回调/辅助逻辑。 | — |
| [`goOnline`](../spatial-studio/src/App.tsx#L103)<br><sub>`goOnline()`</sub> | local function | `App` 中实现`goOnline`的局部回调/辅助逻辑。 | — |
| [`goOffline`](../spatial-studio/src/App.tsx#L104)<br><sub>`goOffline()`</sub> | local function | `App` 中实现`goOffline`的局部回调/辅助逻辑。 | — |
| [`applyResponse`](../spatial-studio/src/App.tsx#L124)<br><sub>`applyResponse(next: SpatialJobResponse, sourceFile: File \| null)`</sub> | local function | `App` 中应用响应的局部回调/辅助逻辑。 | — |
| [`markChanged`](../spatial-studio/src/App.tsx#L143)<br><sub>`markChanged()`</sub> | local function | `App` 中实现`markChanged`的局部回调/辅助逻辑。 | — |
| [`handleUpload`](../spatial-studio/src/App.tsx#L148)<br><sub>`handleUpload(file: File)`</sub> | local function | `App` 中处理`handleUpload`的局部回调/辅助逻辑。 | — |
| [`handleOptions`](../spatial-studio/src/App.tsx#L169)<br><sub>`handleOptions(next: PlanningOptions)`</sub> | local function | `App` 中处理`handleOptions`的局部回调/辅助逻辑。 | — |
| [`handleCoordinate`](../spatial-studio/src/App.tsx#L174)<br><sub>`handleCoordinate(kind: 'start' \| 'goal', axis: 0 \| 1, value: number)`</sub> | local function | `App` 中处理`handleCoordinate`的局部回调/辅助逻辑。 | — |
| [`handleMapPoint`](../spatial-studio/src/App.tsx#L187)<br><sub>`handleMapPoint(point: Coordinate)`</sub> | local function | `App` 中处理`handleMapPoint`的局部回调/辅助逻辑。 | — |
| [`handlePlan`](../spatial-studio/src/App.tsx#L198)<br><sub>`handlePlan()`</sub> | local function | `App` 中处理计划的局部回调/辅助逻辑。 | — |
| [`focusMap`](../spatial-studio/src/App.tsx#L243)<br><sub>`focusMap()`</sub> | local function | `App` 中实现`focusMap`的局部回调/辅助逻辑。 | — |
| [`inspectImage`](../spatial-studio/src/App.tsx#L359)<br><sub>`inspectImage(file: File)`</sub> | function/component | 模块级检查图像的公开操作。 | — |
| [`errorMessage`](../spatial-studio/src/App.tsx#L375)<br><sub>`errorMessage(reason: unknown, fallback: string)`</sub> | function/component | 模块级实现`errorMessage`的公开操作。 | — |

## `spatial-studio/src/api/spatialApi.ts`

共 7 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`constructor`](../spatial-studio/src/api/spatialApi.ts#L14)<br><sub>`constructor(error: StudioError, status: number)`</sub> | constructor | 初始化 `spatialApi` 中定义的 TypeScript 类实例。 | — |
| [`apiUrl`](../spatial-studio/src/api/spatialApi.ts#L22)<br><sub>`apiUrl(path: string)`</sub> | function/component | 模块级实现`apiUrl`的公开操作。 | — |
| [`requestJson`](../spatial-studio/src/api/spatialApi.ts#L27)<br><sub>`requestJson(path: string, init?: RequestInit)`</sub> | function/component | 模块级实现请求的公开操作。 | — |
| [`fetchCapabilities`](../spatial-studio/src/api/spatialApi.ts#L47)<br><sub>`fetchCapabilities(signal?: AbortSignal)`</sub> | function/component | 模块级实现`fetchCapabilities`的公开操作。 | — |
| [`fetchDemo`](../spatial-studio/src/api/spatialApi.ts#L51)<br><sub>`fetchDemo(signal?: AbortSignal)`</sub> | function/component | 模块级实现`fetchDemo`的公开操作。 | — |
| [`planSpatialImage`](../spatial-studio/src/api/spatialApi.ts#L55)<br><sub>`planSpatialImage( file: File, request: SpatialPlanRequest, signal?: AbortSignal, )`</sub> | function/component | 模块级规划计划、图像的公开操作。 | — |
| [`responseImageAsFile`](../spatial-studio/src/api/spatialApi.ts#L70)<br><sub>`responseImageAsFile(response: SpatialJobResponse)`</sub> | function/component | 模块级实现响应、图像的公开操作。 | — |

## `spatial-studio/src/components/AppHeader.tsx`

共 1 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`AppHeader`](../spatial-studio/src/components/AppHeader.tsx#L14)<br><sub>`AppHeader({ response, onUpload, onSettings, onHelp }: AppHeaderProps)`</sub> | function/component | 模块级实现`AppHeader`的公开操作。 | — |

## `spatial-studio/src/components/InspectorPanel.tsx`

共 3 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`InspectorPanel`](../spatial-studio/src/components/InspectorPanel.tsx#L32)<br><sub>`InspectorPanel({ response, selectedRoute, tab, expanded, onSelectRoute, onTab, onExpanded, }: InspectorPanelProps)`</sub> | function/component | 模块级实现`InspectorPanel`的公开操作。 | — |
| [`Metric`](../spatial-studio/src/components/InspectorPanel.tsx#L201)<br><sub>`Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string })`</sub> | function/component | 模块级实现`Metric`的公开操作。 | — |
| [`EmptyInspector`](../spatial-studio/src/components/InspectorPanel.tsx#L211)<br><sub>`EmptyInspector({ title, detail }: { title: string; detail: string })`</sub> | function/component | 模块级实现`EmptyInspector`的公开操作。 | — |

## `spatial-studio/src/components/LayerControls.tsx`

共 1 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`LayerControls`](../spatial-studio/src/components/LayerControls.tsx#L18)<br><sub>`LayerControls({ layers, hasPlan, onChange }: LayerControlsProps)`</sub> | function/component | 模块级实现`LayerControls`的公开操作。 | — |

## `spatial-studio/src/components/LogoMark.tsx`

共 1 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`LogoMark`](../spatial-studio/src/components/LogoMark.tsx#L5)<br><sub>`LogoMark({ size = 38 }: LogoMarkProps)`</sub> | function/component | 模块级实现`LogoMark`的公开操作。 | — |

## `spatial-studio/src/components/MapCanvas.tsx`

共 6 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MapCanvas`](../spatial-studio/src/components/MapCanvas.tsx#L37)<br><sub>`MapCanvas({ image, response, layers, selectedRoute, start, goal, placementMode, stale, onLayers, onSelectRoute, onPoint, onPlacementMode, }: MapCanvasProps)`</sub> | function/component | 模块级实现`MapCanvas`的公开操作。 | — |
| [`worldForecastPolyline`](../spatial-studio/src/components/MapCanvas.tsx#L70)<br><sub>`worldForecastPolyline()`</sub> | local function | `MapCanvas` 中实现世界的局部回调/辅助逻辑。 | — |
| [`orderedRoutes`](../spatial-studio/src/components/MapCanvas.tsx#L80)<br><sub>`orderedRoutes()`</sub> | local function | `MapCanvas` 中实现`orderedRoutes`的局部回调/辅助逻辑。 | — |
| [`pointerCoordinate`](../spatial-studio/src/components/MapCanvas.tsx#L85)<br><sub>`pointerCoordinate(event: React.PointerEvent<SVGSVGElement>)`</sub> | local function | `MapCanvas` 中实现`pointerCoordinate`的局部回调/辅助逻辑。 | — |
| [`handleCanvasPointer`](../spatial-studio/src/components/MapCanvas.tsx#L105)<br><sub>`handleCanvasPointer(event: React.PointerEvent<SVGSVGElement>)`</sub> | local function | `MapCanvas` 中处理`handleCanvasPointer`的局部回调/辅助逻辑。 | — |
| [`MapPoint`](../spatial-studio/src/components/MapCanvas.tsx#L252)<br><sub>`MapPoint({ point, kind, label, radius, fontSize, }: { point: Coordinate kind: 'start' \| 'goal' label: string radius: number fontSize: number })`</sub> | function/component | 模块级实现`MapPoint`的公开操作。 | — |

## `spatial-studio/src/components/MobileCommandBar.tsx`

共 1 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`MobileCommandBar`](../spatial-studio/src/components/MobileCommandBar.tsx#L9)<br><sub>`MobileCommandBar({ onUpload, onPoints, onSettings }: MobileCommandBarProps)`</sub> | function/component | 模块级实现`MobileCommandBar`的公开操作。 | — |

## `spatial-studio/src/components/NavRail.tsx`

共 1 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`NavRail`](../spatial-studio/src/components/NavRail.tsx#L12)<br><sub>`NavRail({ onUpload, onSettings, onInspectorTab, onFocusMap }: NavRailProps)`</sub> | function/component | 模块级实现`NavRail`的公开操作。 | — |

## `spatial-studio/src/components/OverlayDialog.tsx`

共 2 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`OverlayDialog`](../spatial-studio/src/components/OverlayDialog.tsx#L11)<br><sub>`OverlayDialog({ open, title, children, onClose }: OverlayDialogProps)`</sub> | function/component | 模块级实现`OverlayDialog`的公开操作。 | — |
| [`handler`](../spatial-studio/src/components/OverlayDialog.tsx#L14)<br><sub>`handler(event: KeyboardEvent)`</sub> | local function | `OverlayDialog` 中实现`handler`的局部回调/辅助逻辑。 | — |

## `spatial-studio/src/components/StatusBar.tsx`

共 1 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`StatusBar`](../spatial-studio/src/components/StatusBar.tsx#L13)<br><sub>`StatusBar({ response, planning, online, stale, error }: StatusBarProps)`</sub> | function/component | 模块级实现`StatusBar`的公开操作。 | — |

## `spatial-studio/src/components/UploadDialog.tsx`

共 3 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`UploadDialog`](../spatial-studio/src/components/UploadDialog.tsx#L13)<br><sub>`UploadDialog({ open, maxUploadMb, onClose, onFile }: UploadDialogProps)`</sub> | function/component | 模块级实现`UploadDialog`的公开操作。 | — |
| [`handleKey`](../spatial-studio/src/components/UploadDialog.tsx#L20)<br><sub>`handleKey(event: KeyboardEvent)`</sub> | local function | `UploadDialog` 中处理`handleKey`的局部回调/辅助逻辑。 | — |
| [`accept`](../spatial-studio/src/components/UploadDialog.tsx#L29)<br><sub>`accept(file: File \| undefined)`</sub> | local function | `UploadDialog` 中实现`accept`的局部回调/辅助逻辑。 | — |

## `spatial-studio/src/components/WorkflowPanel.tsx`

共 2 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`CoordinateControl`](../spatial-studio/src/components/WorkflowPanel.tsx#L26)<br><sub>`CoordinateControl({ label, point, active, color, onActivate, onChange, }: { label: string point: Coordinate active: boolean color: 'green' \| 'red' onActivate: () => void onChan…`</sub> | function/component | 模块级实现`CoordinateControl`的公开操作。 | — |
| [`WorkflowPanel`](../spatial-studio/src/components/WorkflowPanel.tsx#L82)<br><sub>`WorkflowPanel({ imageName, start, goal, placementMode, options, capabilities, planning, compact = false, onUpload, onPlacementMode, onCoordinate, onOptions, onPlan, }: Work…`</sub> | function/component | 模块级实现`WorkflowPanel`的公开操作。 | — |

## `spatial-studio/src/lib/preferences.ts`

共 2 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`loadPreferences`](../spatial-studio/src/lib/preferences.ts#L34)<br><sub>`loadPreferences()`</sub> | function/component | 模块级加载`loadPreferences`的公开操作。 | — |
| [`savePreferences`](../spatial-studio/src/lib/preferences.ts#L60)<br><sub>`savePreferences( layers: LayerVisibility, options: PlanningOptions, )`</sub> | function/component | 模块级保存`savePreferences`的公开操作。 | — |

## `spatial-studio/src/lib/routes.ts`

共 3 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`routeStyle`](../spatial-studio/src/lib/routes.ts#L12)<br><sub>`routeStyle(index: number)`</sub> | function/component | 模块级实现路线的公开操作。 | — |
| [`routeDisplayName`](../spatial-studio/src/lib/routes.ts#L16)<br><sub>`routeDisplayName(route: SpatialRoute, index: number)`</sub> | function/component | 模块级实现路线的公开操作。 | — |
| [`routePolyline`](../spatial-studio/src/lib/routes.ts#L22)<br><sub>`routePolyline(route: SpatialRoute)`</sub> | function/component | 模块级实现路线的公开操作。 | — |

## `spatial-studio/src/lib/viewState.ts`

共 2 个具名可调用项。

| 函数 / 方法 | 类型 | 意义 | 静态可见的直接调用（最多 10 个） |
|---|---|---|---|
| [`parseViewState`](../spatial-studio/src/lib/viewState.ts#L10)<br><sub>`parseViewState(search: string)`</sub> | function/component | 模块级解析状态的公开操作。 | — |
| [`writeViewState`](../spatial-studio/src/lib/viewState.ts#L20)<br><sub>`writeViewState(state: ViewState)`</sub> | function/component | 模块级写入状态的公开操作。 | — |

## 生成与校验

```powershell
python tools/generate_function_reference.py
python tools/generate_function_reference.py --check
```

`--check` 不改文件；当生成结果与已提交文档不一致时返回非零退出码。
