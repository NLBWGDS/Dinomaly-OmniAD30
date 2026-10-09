# 1.0.3 平台交付与排障清单

本次按两份 PDF 原文核对并修改的是 Docker 入口、日志、协议验证和打包流程，不修改算法精度策略。
此前版本已有运行阶段日志，但缺少模块导入前的诊断，也没有提供推理 PDF 示例中的目录。
这两处已修复；没有平台实际启动命令和 Docker 错误，仍不能断言之前任务的根因。

## 1. 同步源码后在服务器执行

请同步整个项目源码，尤其不要漏掉新增的 `platform_bootstrap.py`、`platform_launch.sh`、
`verify_platform_startup.py`，以及修改后的两个 Dockerfile、入口脚本、适配器和验证脚本。
本地改代码不会更新已经上传的镜像。保留旧版本，不用删除 1.0.1/1.0.2。

在仍有 `omniad_dinomaly:1.0.1` 的 Ubuntu 服务器项目目录执行：

```bash
cd ~/xzh/IAD_Competition/Dinomaly-OmniAD30
bash package_platform_release.sh \
  ../dataset/download/Omni-AD-30-release/work_piece14 \
  "$HOME/omniad_release_1.0.3"
```

输出目录必须尚不存在。脚本复用 1.0.1 的依赖环境，不重新下载模型。
依次构建新镜像、检查 7 个预期失败场景的日志、运行 100 epoch 真实 GPU 训练、
对该类别本地 test 中的所有图片推理、检查结果协议、导出并重新加载 TAR、验证 ZIP。
正常训练和推理均使用文档示例入口，禁用容器联网。
任一步失败都会停止，不生成“可交付成功”的提示；日志保留在发布目录，不要忽略错误继续上传。
这些验证是运行和格式验证，不是 30 类精度评估，也不代替平台实际测试。

成功结束应看到：

```text
Release ready: /home/ubuntu/omniad_release_1.0.3/omniad_dinomaly_1.0.3.zip
```

上传文件是该 ZIP；平台镜像版本填写：

```text
zhejiang_ai_competition:hikrobot_comp_40_v3
```

下载完成后，在 Windows PowerShell 计算 `Get-FileHash -Algorithm SHA256 "本地ZIP完整路径"`，
结果必须与发布目录 `SHA256SUMS` 相同，再上传。不要将旧 ZIP 仅改名为新版本。

## 2. 交给平台工程师核对的入口

推理库配置为非驻留，算法类型 103，选择新的 v3 版本。推理启动命令模板：

```bash
docker run --runtime=nvidia --cap-add=ALL \
  --env NVIDIA_VISIBLE_DEVICES=${visibleDevice} \
  --name=${taskName} \
  -v /tmp/:/tmp/ \
  -v ${inputPath}:/input \
  -v ${outputPath}:/output \
  --rm ${tagName} /bin/bash \
  -c 'cd /home/apps/ats-reasoning-tool/ && sh start.sh /input/ /output/'
```

训练启动命令模板：

```bash
docker run --cap-add=ALL \
  --gpus '"device=${gpuNumber}"' \
  --name=${containerName} \
  --security-opt seccomp=unconfined \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /dev/shm:/dev/shm \
  -v ${inputDir}:/input \
  -v ${outputDir}:/output \
  --entrypoint /bin/bash --rm ${imageVersion} \
  -c "cd ./root && python train.py"
```

`${...}` 是平台后台替换的变量，不是在个人电脑直接执行的现成命令。
镜像无 ENTRYPOINT，默认 CMD 是推理；训练库必须配置训练命令。
推理 PDF 第 4 页明确说明程序启动命令在后台固定配置，上传新镜像并不会自动修改后台配置。

## 3. 日志应该在哪里

以下容器路径对应本任务 `${outputPath}` / `${outputDir}` 所挂载的宿主机目录：

| 文件 | 内容 |
| --- | --- |
| `/output/reasoning.log` | 推理开始、每图耗时、标准错误或成功结束 |
| `/output/state.txt` | 训练实时 epoch/iter/lr/eta/time/memory/loss；模型导出成功后才写 finish |
| `/output/bootstrap.log` | Python 启动版本、加载阶段、stdout/stderr、完整异常堆栈、退出码 |
| `/output/entrypoint.log` | 通过 start.sh/train.sh 启动时的控制台输出；包括 Python 无法启动 |
| `/output/train_debug.log` | 训练子进程输出及异常；只有进入训练适配器后才创建 |

标准日志使用 UTF-8，实时刷新，同步输出控制台。原文要求的开始和成功标识保持原样。
额外诊断放在独立日志，不在 `state.txt` 错误内容里误写 `finish`。

## 4. 如果仍然提示“镜像启动失败，未输出日志”

请工程师提供这些信息，而不是只检查算法日志文件：

1. 本任务实际使用的完整镜像 tag/ID，以及模板变量替换后的 Docker 命令（敏感值可打码）。
2. 执行 `docker run` 本身的 stdout、stderr 和退出码。
3. 容器若已创建，提供 `docker inspect <容器名>` 的 State、Mounts、Path、Args，以及 `docker logs <容器名>`。
4. 该任务真实输出挂载目录中的上述日志。确认看的是任务所在节点，而非其他机器或空目录。

调试时可由工程师去掉 `--rm` 保留失败容器；正常上线仍使用文档的自动删除方式。
没有创建容器、GPU runtime 拒绝启动、执行入口前 cd 失败等情况，镜像内程序尚未运行，不能自行写日志。
输出目录不可写时只能向控制台报错。此时应读取 Docker 的启动错误，不应通过反复重打包猜测原因。

## 5. 交付边界

- 本地单元/子进程测试可验证适配和错误日志，但当前本机未安装 Docker，未在此电脑构建新镜像。
- 新版需要在服务器完成脚本验证，再在平台验证；不能把先前 1.0.2 的成功记录当成 1.0.3 已通过。
- 模型加密格式未提供，因此不支持加密模型；`modelPassword` 留空。
- `gt.json` 按推理文档由评估工具生成，适配器不编造真值、不覆盖已有真值。
- 当前平台适配仍为基础 Dinomaly 路径，不含开发评估的分类别 memory bank/特殊推理路由。
