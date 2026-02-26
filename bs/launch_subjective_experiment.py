#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主观评价实验启动脚本
一键启动完整的实验系统
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def print_header():
    """打印标题"""
    print("="*60)
    print("🧪 AI生成说话人脸视频主观评价实验系统")
    print("="*60)
    print()

def print_steps():
    """显示实验步骤"""
    print("📋 实验设置步骤:")
    print("1. 🔍 检查环境和依赖")
    print("2. 📊 运行实验设计")
    print("3. 🎬 运行视频选取")
    print("4. 🌐 启动Web服务器")
    print("5. 📋 提供操作指南")
    print()

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 6:
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        return False
    
    # 检查必要的库
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn', 'flask'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少必要的包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    # 检查必要的文件
    required_files = [
        'config/optimized_config.yaml',
        'datasets/ac.pkl'
       
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ 缺少必要的文件: {', '.join(missing_files)}")
        print("请确保:")
        print("1. 配置文件存在")
        print("2. 数据集文件存在")
        print("3. 模型训练已完成并保存了检查点")
        return False
    
    print("✅ 环境检查通过")
    return True

def run_experiment_design():
    """运行实验设计"""
    print("\n📊 运行实验设计...")
    
    try:
        result = subprocess.run([sys.executable, 'subjective_evaluation_design.py'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ 实验设计完成")
            print(result.stdout)
            return True
        else:
            print(f"❌ 实验设计失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 实验设计超时")
        return False
    except Exception as e:
        print(f"❌ 运行实验设计时出错: {e}")
        return False

def run_video_selection():
    """运行视频选取"""
    print("\n🎬 运行视频选取...")
    
    try:
        result = subprocess.run([sys.executable, 'video_selection_manager.py'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ 视频选取完成")
            print(result.stdout)
            return True
        else:
            print(f"❌ 视频选取失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 视频选取超时")
        return False
    except Exception as e:
        print(f"❌ 运行视频选取时出错: {e}")
        return False

def start_web_server():
    """启动Web服务器"""
    print("\n🌐 启动Web服务器...")
    
    try:
        # 在后台启动服务器
        server_process = subprocess.Popen([sys.executable, 'subjective_experiment_server.py'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        
        # 等待服务器启动
        time.sleep(3)
        
        # 检查服务器是否正常运行
        if server_process.poll() is None:
            print("✅ Web服务器启动成功")
            print("📱 服务器地址: http://localhost:5000")
            print("🔧 管理后台: http://localhost:5000/admin")
            return server_process
        else:
            print("❌ Web服务器启动失败")
            return None
    except Exception as e:
        print(f"❌ 启动Web服务器时出错: {e}")
        return None

def provide_instructions():
    """提供操作指南"""
    print("\n📋 操作指南")
    print("="*60)
    print()
    
    print("🎯 实验参与流程:")
    print("1. 参与者访问: http://localhost:5000")
    print("2. 阅读实验说明并同意参与")
    print("3. 填写基本信息")
    print("4. 了解评价指导")
    print("5. 进行练习评价")
    print("6. 完成20个视频对评价")
    print("7. 获得实验完成确认")
    print()
    
    print("🔧 管理员功能:")
    print("- 管理后台: http://localhost:5000/admin")
    print("- 查看实时统计信息")
    print("- 导出评价数据")
    print("- 监控实验进展")
    print()
    
    print("📁 重要文件位置:")
    print("- 实验设计: subjective_experiment/experiment_design.json")
    print("- 视频选取: subjective_experiment/selection_results.json")
    print("- 评价数据: subjective_experiment/evaluation_results.csv")
    print("- 参与者信息: subjective_experiment/participants.json")
    print()
    
    print("⚠️ 注意事项:")
    print("1. 确保视频文件准备完毕")
    print("2. 测试评价功能正常工作")
    print("3. 定期备份实验数据")
    print("4. 监控数据质量")
    print()
    
    print("📊 数据分析建议:")
    print("1. 完成实验后运行统计分析")
    print("2. 对比原始模型和优化模型的评分")
    print("3. 分析评价者间一致性")
    print("4. 生成可视化报告")
    print()

def generate_quick_start_script():
    """生成快速启动脚本"""
    
    script_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速启动主观评价实验
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def quick_start():
    """快速启动实验"""
    
    print("🚀 快速启动主观评价实验...")
    
    # 检查主观实验目录
    if not Path("subjective_experiment").exists():
        print("📊 正在设置实验...")
        subprocess.run([sys.executable, "subjective_evaluation_design.py"])
        subprocess.run([sys.executable, "video_selection_manager.py"])
    
    # 启动服务器
    print("🌐 启动Web服务器...")
    subprocess.run([sys.executable, "subjective_experiment_server.py"])

if __name__ == "__main__":
    quick_start()
'''
    
    with open('quick_start_experiment.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod('quick_start_experiment.py', 0o755)
    print("✅ 快速启动脚本已创建: quick_start_experiment.py")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动主观评价实验系统")
    parser.add_argument('--skip-check', action='store_true', help='跳过环境检查')
    parser.add_argument('--skip-design', action='store_true', help='跳过实验设计')
    parser.add_argument('--skip-selection', action='store_true', help='跳过视频选取')
    parser.add_argument('--server-only', action='store_true', help='仅启动服务器')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    
    args = parser.parse_args()
    
    print_header()
    print_steps()
    
    # 环境检查
    if not args.skip_check and not args.server_only:
        if not check_environment():
            print("❌ 环境检查失败，请解决上述问题后重试")
            return
    
    # 实验设计
    if not args.skip_design and not args.server_only:
        if not run_experiment_design():
            print("❌ 实验设计失败")
            return
    
    # 视频选取
    if not args.skip_selection and not args.server_only:
        if not run_video_selection():
            print("❌ 视频选取失败")
            return
    
    # 启动服务器
    server_process = start_web_server()
    if server_process is None:
        print("❌ 无法启动Web服务器")
        return
    
    # 生成快速启动脚本
    generate_quick_start_script()
    
    # 提供操作指南
    provide_instructions()
    
    print("🎉 实验系统启动完成！")
    print()
    print("💡 提示:")
    print("- 按 Ctrl+C 停止服务器")
    print("- 编辑 quick_start_experiment.py 进行自定义配置")
    print("- 查看生成的 subjective_experiment 目录了解实验结构")
    
    try:
        # 等待用户中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务器...")
        server_process.terminate()
        server_process.wait()
        print("✅ 服务器已停止")

if __name__ == "__main__":
    main()