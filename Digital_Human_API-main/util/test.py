import os, win32com.client as wc, pythoncom
ppt_path = r"F:\hlh\test.ppt" # 改成你的绝对路径
print("path:", ppt_path)
print("exists:", os.path.exists(ppt_path), "isfile:", os.path.isfile(ppt_path))

pythoncom.CoInitialize()
app = wc.Dispatch("PowerPoint.Application")
app.Visible = True
try:
  pres = app.Presentations.Open(ppt_path, False, False, True)
  print("Opened OK")
  pres.Close()
finally:
  app.Quit()
pythoncom.CoUninitialize()