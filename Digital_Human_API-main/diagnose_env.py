import sys, os, importlib, json, platform
print('--- 环境诊断 ---')
print('Python:', sys.version)
print('Platform:', platform.platform())
print('CWD:', os.getcwd())
print('\n前10个 sys.path:')
for i,p in enumerate(sys.path[:10]):
    print(f'{i}: {p}')

result = {}
for mod in ['GPT_SoVITS.inference_webui','LangSegment']:
    try:
        m = importlib.import_module(mod)
        result[mod] = getattr(m,'__file__','<no __file__>')
    except Exception as e:
        result[mod] = f'导入失败: {e}'
print('\n模块路径:')
print(json.dumps(result, ensure_ascii=False, indent=2))

print('\n若 inference_webui 路径 不是 当前项目下的 VITS/GPT_SoVITS/inference_webui.py，说明存在重复副本。')
print('可在 server1.4.0.py 顶部加入:')
print('    import sys, os; sys.path.insert(0, os.path.dirname(__file__))')
