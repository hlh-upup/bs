# 本地替换有问题的第三方 LangSegment 版本，提供最小接口
# 功能：按中文/英文/日文/韩文粗略切分；当前仅简单实现，不做复杂模型判断。

_supported = {"zh","en","ja","ko"}
_current_filters = list(_supported)

def setfilters(filters):
    global _current_filters
    if not filters:
        _current_filters = list(_supported)
    else:
        _current_filters = [f for f in filters if f in _supported]

# 兼容一些源码里可能引用的别名（万一）
setLangfilters = setfilters  # type: ignore

def getfilters():
    return _current_filters

def getTexts(text: str):
    # 简单策略：按空白分词，英文 token 标记 en，其它统一 zh；没有复杂语言检测
    parts = text.strip().split()
    if not parts:
        return []
    out = []
    for token in parts:
        if all(ord(c) < 128 for c in token):
            lang = "en" if "en" in _current_filters else (_current_filters[0] if _current_filters else "en")
        else:
            lang = "zh" if "zh" in _current_filters else (_current_filters[0] if _current_filters else "zh")
        out.append({"text": token, "lang": lang})
    return out

# 兼容源码里可能调用的占位函数
def getCounts(text: str):
    return len(getTexts(text))

def classify(text: str):
    # 返回最先匹配到的过滤语言
    items = getTexts(text)
    if not items:
        return None
    return items[0]["lang"]

print("[Local LangSegment stub loaded]")
