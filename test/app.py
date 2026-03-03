from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd
import random

app = Flask(__name__)

# 视频文件夹路径
VIDEO_FOLDER = os.path.join(app.root_path, 'static', 'videos')
USER_INFO = {"name": "", "age": "", "selected_folder": ""}


@app.route('/')
def index():
    # 显示文件夹选择和用户信息输入
    subfolders = [f.name for f in os.scandir(VIDEO_FOLDER) if f.is_dir()]
    return render_template('index.html', subfolders=subfolders)


@app.route('/select_folder', methods=['POST'])
def select_folder():
    # 获取选择的文件夹和用户信息
    selected_folder = request.form.get('folder')
    USER_INFO['name'] = request.form.get('name')
    USER_INFO['age'] = request.form.get('age')
    USER_INFO['selected_folder'] = selected_folder

    # 获取子文件夹中的视频文件
    video_path = os.path.join(VIDEO_FOLDER, selected_folder)
    video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
    random.shuffle(video_files)  # 打乱视频顺序

    # 将视频文件名传递给前端
    return render_template(
        'videos.html',
        folder=selected_folder,
        video_files=video_files,
        user_info=USER_INFO
    )

@app.route('/videos/<folder>/<filename>')
def videos(folder, filename):
    # 返回视频文件
    video_path = os.path.join(VIDEO_FOLDER, folder)
    return send_from_directory(video_path, filename)


@app.route('/submit_ratings', methods=['POST'])
def submit_ratings():
    # 获取视频数量
    video_count = int(request.form.get('video_count', 0))

    # 获取用户信息（从全局变量）
    username = USER_INFO['name']
    age = USER_INFO['age']
    folder_name = USER_INFO['selected_folder']

    # 初始化评分数据
    ratings = {
        '视频名称': [],  # 修改为存储文件名
        '口型同步效果': [],
        '表情自然度': [],
        '总体评价': [],
        '音频质量': [],
        '跨模态一致性': [],
    }

    # 循环获取评分
    for i in range(1, video_count + 1):
        video_name = request.form.get(f'video_name_{i}')  # 获取视频文件名
        ratings['视频名称'].append(video_name)
        ratings['口型同步效果'].append(request.form.get(f'lip_sync_{i}'))
        ratings['表情自然度'].append(request.form.get(f'expression_naturalness_{i}'))
        ratings['总体评价'].append(request.form.get(f'overall_{i}'))
        ratings['音频质量'].append(request.form.get(f'audio_quality_{i}'))
        ratings['跨模态一致性'].append(request.form.get(f'cross_modal_consistency_{i}'))

    # 保存到 Excel 文件，命名为 "文件夹名_用户名_年龄.xlsx"
    file_name = f"{folder_name}_{username}_{age}.xlsx"
    file_path = os.path.join(app.root_path, file_name)

    df = pd.DataFrame(ratings)
    try:
        df.to_excel(file_path, index=False)
    except Exception as e:
        return f"Error saving file: {str(e)}"

    # 返回保存成功页面，传递二维码图片
    return render_template(
        'success.html',
        message="评分已成功保存！",
        qr_image_url=url_for('static', filename='问卷.jpg')
    )


if __name__ == '__main__':
    app.run(debug=True)

    app.run(debug=True)
