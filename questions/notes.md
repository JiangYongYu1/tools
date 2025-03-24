# Questions
## 1. git pull出现如下问题，按 ctrl+X退出 ![question 1](../asserts/images/git_merge.png)
## 2. nv的bt601 yuv和rgb的转换 ![Alt text](../asserts/images/nv_rgb_yuv.png)
## 3. vscode debug时出现一串错误，但是能继续跑，这个大多数是由于参赛解析错误导致的，主要使用 ```pip install --upgrade importlib-metadata```可以解决
    ```shell
        usr/bin/env /home/jiangyong.yu/miniconda3/envs/qwen2vl/bin/python /home/jiangyong.yu/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/adapter/../../
    debugpy/launcher 49657 -- /home/jiangyong.yu/work/work/qwen2.5-vl/demo.py 
    E+00000.023: Error while enumerating installed packages.
                
                Traceback (most recent call last):
                File "/home/jiangyong.yu/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/common/log.py", line 361, in get_environment_description
                    report("    {0}=={1}\n", pkg.name, pkg.version)
                AttributeError: 'PathDistribution' object has no attribute 'name'
                
                Stack where logged:
                File "/home/jiangyong.yu/miniconda3/envs/qwen2vl/lib/python3.9/runpy.py", line 197, in _run_module_as_main
                    return _run_code(code, main_globals, None,
                File "/home/jiangyong.yu/miniconda3/envs/qwen2vl/lib/python3.9/runpy.py", line 87, in _run_code
                    exec(code, run_globals)
                File "/home/jiangyong.yu/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/__main__.py", line 91, in <module>
                    main()
                File "/home/jiangyong.yu/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/__main__.py", line 21, in main
                    log.describe_environment("debugpy.launcher startup environment:")
                File "/home/jiangyong.yu/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/common/log.py", line 369, in describe_environment
                    info("{0}", get_environment_description(header))
                File "/home/jiangyong.yu/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/common/log.py", line 363, in get_environment_description
                    swallow_exception("Error while enumerating installed packages.")
                File "/home/jiangyong.yu/.vscode-server/extensions/ms-python.debugpy-2024.0.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/common/log.py", line 215, in swallow_exception
                    _exception(format_string, *args, **kwargs)
                

    E+00000.016: Error while enumerating installed packages.
    ```
## 4. 配置ssh key后，git clone github报错，如下：
    ```shell
    git clone git@github.com:JiangYongYu1/fp16lut.git
    Cloning into 'fp16lut'...
    kex_exchange_identification: Connection closed by remote host
    fatal: Could not read from remote repository.
    ```
    解决方法: 
    ```shell
    ssh -vT git@github.com, ssh -T git@github.com都可以如果配置正确，你应该看到类似 “Hi username! You’ve successfully authenticated…” 的提示。如果出现错误信息，可以根据调试信息进一步排查。
    GitHub 支持通过 443 端口进行 SSH 连接。如果 22 端口受限，可以修改 SSH 客户端配置。编辑（或创建） ~/.ssh/config 文件，添加如下配置：
    ```
    ```
    Host github.com
        Hostname ssh.github.com
        Port 443
    ```
    配置后问题解决