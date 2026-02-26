#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主观评价实验服务器
提供Web界面用于收集主观评价数据
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import uuid
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

class SubjectiveExperimentServer:
    """主观评价实验服务器"""
    
    def __init__(self, experiment_dir: str = 'subjective_experiment'):
        self.experiment_dir = Path(experiment_dir)
        self.data_file = self.experiment_dir / 'experiment_data.json'
        self.results_file = self.experiment_dir / 'evaluation_results.csv'
        self.participants_file = self.experiment_dir / 'participants.json'
        
        # 加载实验数据
        self.experiment_data = self.load_experiment_data()
        self.participants = self.load_participants()
        
        # 确保目录存在
        self.experiment_dir.mkdir(exist_ok=True)
        
    def load_experiment_data(self) -> Dict:
        """加载实验数据"""
        if self.data_file.exists():
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_participants(self) -> Dict:
        """加载参与者数据"""
        if self.participants_file.exists():
            with open(self.participants_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_participants(self):
        """保存参与者数据"""
        with open(self.participants_file, 'w', encoding='utf-8') as f:
            json.dump(self.participants, f, ensure_ascii=False, indent=2, default=str)
    
    def generate_participant_id(self) -> str:
        """生成唯一参与者ID"""
        return f"P_{uuid.uuid4().hex[:8].upper()}"
    
    def create_participant_session(self, participant_id: str) -> Dict:
        """创建参与者会话"""
        session = {
            'participant_id': participant_id,
            'created_time': datetime.now().isoformat(),
            'current_trial': 0,
            'completed_trials': [],
            'responses': [],
            'status': 'started',
            'demographics': {}
        }
        
        self.participants[participant_id] = session
        self.save_participants()
        
        return session
    
    def get_next_trial(self, participant_id: str) -> Optional[Dict]:
        """获取下一个试验"""
        if participant_id not in self.participants:
            return None
            
        participant = self.participants[participant_id]
        current_trial = participant['current_trial']
        
        if current_trial >= len(self.experiment_data.get('trials', [])):
            return None
            
        trial = self.experiment_data['trials'][current_trial]
        trial['trial_index'] = current_trial
        
        return trial
    
    def save_response(self, participant_id: str, trial_data: Dict, responses: Dict):
        """保存评价响应"""
        if participant_id not in self.participants:
            return False
            
        participant = self.participants[participant_id]
        
        # 创建响应记录
        response_record = {
            'participant_id': participant_id,
            'trial_id': trial_data['trial_id'],
            'trial_index': trial_data['trial_index'],
            'timestamp': datetime.now().isoformat(),
            'responses': responses,
            'video_pair': trial_data['video_pair'],
            'presentation_order': trial_data['presentation_order']
        }
        
        participant['responses'].append(response_record)
        participant['completed_trials'].append(trial_data['trial_id'])
        participant['current_trial'] += 1
        
        # 检查是否完成所有试验
        if participant['current_trial'] >= len(self.experiment_data.get('trials', [])):
            participant['status'] = 'completed'
            participant['completion_time'] = datetime.now().isoformat()
        
        self.save_participants()
        
        # 保存到CSV文件
        self.save_response_to_csv(response_record)
        
        return True
    
    def save_response_to_csv(self, response_record: Dict):
        """保存响应到CSV文件"""
        # 准备数据行
        row = {
            'participant_id': response_record['participant_id'],
            'trial_id': response_record['trial_id'],
            'trial_index': response_record['trial_index'],
            'timestamp': response_record['timestamp'],
            'video_index': response_record['video_pair']['video_index']
        }
        
        # 添加评分数据
        responses = response_record['responses']
        for key, value in responses.items():
            if key != 'comments':
                row[key] = value
        
        # 添加评论
        row['comments'] = responses.get('comments', '')
        
        # 读取或创建CSV文件
        if self.results_file.exists():
            df = pd.read_csv(self.results_file)
            df_new = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df_new = pd.DataFrame([row])
        
        # 保存CSV文件
        df_new.to_csv(self.results_file, index=False)
    
    def get_experiment_statistics(self) -> Dict:
        """获取实验统计信息"""
        total_participants = len(self.participants)
        completed_participants = sum(1 for p in self.participants.values() if p['status'] == 'completed')
        
        # 计算平均完成时间
        completion_times = []
        for p in self.participants.values():
            if p['status'] == 'completed' and 'completion_time' in p:
                start_time = datetime.fromisoformat(p['created_time'])
                end_time = datetime.fromisoformat(p['completion_time'])
                duration = (end_time - start_time).total_seconds() / 60  # 分钟
                completion_times.append(duration)
        
        avg_duration = np.mean(completion_times) if completion_times else 0
        
        return {
            'total_participants': total_participants,
            'completed_participants': completed_participants,
            'completion_rate': (completed_participants / total_participants * 100) if total_participants > 0 else 0,
            'average_duration': avg_duration,
            'total_trials': len(self.experiment_data.get('trials', []))
        }

# 创建服务器实例
server = SubjectiveExperimentServer()

@app.route('/')
def index():
    """首页"""
    return redirect(url_for('welcome'))

@app.route('/welcome')
def welcome():
    """欢迎页面"""
    return render_template('welcome.html')

@app.route('/consent')
def consent():
    """知情同意页面"""
    return render_template('consent.html')

@app.route('/demographics', methods=['GET', 'POST'])
def demographics():
    """人口统计学信息收集"""
    if request.method == 'POST':
        participant_id = server.generate_participant_id()
        demographics_data = request.form.to_dict()
        
        # 创建参与者会话
        session = server.create_participant_session(participant_id)
        session['demographics'] = demographics_data
        
        return redirect(url_for('instructions', participant_id=participant_id))
    
    return render_template('demographics.html')

@app.route('/instructions/<participant_id>')
def instructions(participant_id):
    """指导页面"""
    return render_template('instructions.html', participant_id=participant_id)

@app.route('/practice/<participant_id>')
def practice(participant_id):
    """练习页面"""
    return render_template('practice.html', participant_id=participant_id)

@app.route('/experiment/<participant_id>')
def experiment(participant_id):
    """主实验页面"""
    if participant_id not in server.participants:
        return redirect(url_for('index'))
    
    trial = server.get_next_trial(participant_id)
    if not trial:
        # 实验完成
        return redirect(url_for('completion', participant_id=participant_id))
    
    # 获取实验统计信息
    stats = server.get_experiment_statistics()
    
    return render_template('experiment.html', 
                         participant_id=participant_id,
                         trial=trial,
                         trial_number=trial['trial_index'] + 1,
                         total_trials=stats['total_trials'])

@app.route('/completion/<participant_id>')
def completion(participant_id):
    """实验完成页面"""
    if participant_id not in server.participants:
        return redirect(url_for('index'))
    
    participant = server.participants[participant_id]
    
    # 计算实验时长
    start_time = datetime.fromisoformat(participant['created_time'])
    if participant['status'] == 'completed' and 'completion_time' in participant:
        end_time = datetime.fromisoformat(participant['completion_time'])
        duration = (end_time - start_time).total_seconds() / 60
    else:
        duration = 0
    
    return render_template('completion.html', 
                         participant_id=participant_id,
                         duration=duration,
                         completed_trials=len(participant['completed_trials']))

@app.route('/admin')
def admin():
    """管理员界面"""
    stats = server.get_experiment_statistics()
    return render_template('admin.html', stats=stats)

@app.route('/api/submit_evaluation', methods=['POST'])
def submit_evaluation():
    """提交评价数据"""
    try:
        data = request.get_json()
        participant_id = data['participant_id']
        trial_data = data['trial_data']
        responses = data['responses']
        
        success = server.save_response(participant_id, trial_data, responses)
        
        if success:
            return jsonify({'success': True, 'message': '评价提交成功'})
        else:
            return jsonify({'success': False, 'message': '提交失败'})
            
    except Exception as e:
        logger.error(f"提交评价时出错: {e}")
        return jsonify({'success': False, 'message': '服务器错误'})

@app.route('/api/get_next_trial/<participant_id>')
def get_next_trial(participant_id):
    """获取下一个试验"""
    trial = server.get_next_trial(participant_id)
    
    if trial:
        return jsonify({'success': True, 'trial': trial})
    else:
        return jsonify({'success': False, 'message': '所有试验已完成'})

@app.route('/api/get_statistics')
def get_statistics():
    """获取统计信息"""
    stats = server.get_experiment_statistics()
    return jsonify(stats)

@app.route('/api/export_results')
def export_results():
    """导出结果数据"""
    if server.results_file.exists():
        return send_from_directory(server.experiment_dir, 'evaluation_results.csv', 
                                 as_attachment=True)
    else:
        return jsonify({'success': False, 'message': '暂无数据'})

def create_templates():
    """创建HTML模板"""
    
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # 欢迎页面模板
    welcome_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI生成说话人脸视频质量评价</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .info-box { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #007bff; }
        .btn { display: inline-block; padding: 12px 30px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px; }
        .btn:hover { background: #0056b3; }
        .steps { margin: 30px 0; }
        .step { margin: 15px 0; padding: 15px; background: #e9ecef; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 AI生成说话人脸视频质量评价实验</h1>
        
        <div class="info-box">
            <h3>📋 实验说明</h3>
            <p>感谢您参与我们的科学研究实验！本实验旨在评估AI生成的说话人脸视频质量。</p>
            <p>您的真实反馈将帮助我们改进AI技术，为用户提供更好的体验。</p>
        </div>
        
        <div class="steps">
            <h3>🎯 实验流程</h3>
            <div class="step">
                <strong>步骤 1:</strong> 阅读并同意知情同意书
            </div>
            <div class="step">
                <strong>步骤 2:</strong> 填写基本信息
            </div>
            <div class="step">
                <strong>步骤 3:</strong> 了解评价指导
            </div>
            <div class="step">
                <strong>步骤 4:</strong> 进行练习评价
            </div>
            <div class="step">
                <strong>步骤 5:</strong> 正式实验评价
            </div>
            <div class="step">
                <strong>步骤 6:</strong> 完成实验并获得感谢
            </div>
        </div>
        
        <div class="info-box">
            <h3>⏱️ 时间安排</h3>
            <ul>
                <li>预计总时长: 20-30分钟</li>
                <li>练习阶段: 5分钟</li>
                <li>正式实验: 15-25分钟</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h3>🎁 参与获益</h3>
            <ul>
                <li>为AI技术发展做出贡献</li>
                <li>体验前沿的人机交互技术</li>
                <li>获得参与证书（如需要）</li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 30px;">
            <a href="/consent" class="btn">开始实验</a>
        </div>
    </div>
</body>
</html>
    """
    
    # 知情同意页面模板
    consent_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知情同意书</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
        h1 { color: #333; text-align: center; }
        .consent-text { line-height: 1.6; margin: 20px 0; }
        .checkbox-group { margin: 20px 0; }
        .btn { display: inline-block; padding: 12px 30px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px; }
        .btn:hover { background: #0056b3; }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="container">
        <h1>知情同意书</h1>
        
        <div class="consent-text">
            <h3>研究目的</h3>
            <p>本研究旨在评估AI生成的说话人脸视频质量，改进相关技术。</p>
            
            <h3>研究过程</h3>
            <p>您将被要求观看一系列AI生成的视频，并对视频质量进行评价。整个过程约需20-30分钟。</p>
            
            <h3>风险与不适</h3>
            <p>本实验风险很低，可能会因长时间观看屏幕导致轻微眼部疲劳。</p>
            
            <h3>隐私保护</h3>
            <p>您的所有数据将被匿名化处理，我们不会收集任何个人身份信息。</p>
            
            <h3>自愿参与</h3>
            <p>参与本实验完全出于自愿，您可以在任何时候退出实验。</p>
            
            <h3>联系方式</h3>
            <p>如有疑问，请联系研究团队：research@example.com</p>
        </div>
        
        <div class="checkbox-group">
            <label>
                <input type="checkbox" id="consent1" required>
                我已阅读并理解上述信息
            </label><br><br>
            <label>
                <input type="checkbox" id="consent2" required>
                我自愿参与本实验
            </label><br><br>
            <label>
                <input type="checkbox" id="consent3" required>
                我同意研究者使用我的匿名数据
            </label><br><br>
            <label>
                <input type="checkbox" id="consent4" required>
                我知晓可以随时退出实验
            </label>
        </div>
        
        <div style="text-align: center; margin-top: 30px;">
            <button class="btn" id="continueBtn" disabled>继续</button>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]');
            const continueBtn = document.getElementById('continueBtn');
            
            function updateButton() {
                const allChecked = Array.from(checkboxes).every(cb => cb.checked);
                continueBtn.disabled = !allChecked;
            }
            
            checkboxes.forEach(cb => cb.addEventListener('change', updateButton));
            
            continueBtn.addEventListener('click', function() {
                window.location.href = '/demographics';
            });
        });
    </script>
</body>
</html>
    """
    
    # 人口统计学信息模板
    demographics_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>基本信息</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
        h1 { color: #333; text-align: center; }
        .form-group { margin: 20px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .btn { display: inline-block; padding: 12px 30px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px; }
        .btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>基本信息</h1>
        <p>请填写您的基本信息（仅用于研究分析，严格保密）</p>
        
        <form method="POST">
            <div class="form-group">
                <label for="age">年龄:</label>
                <select name="age" id="age" required>
                    <option value="">请选择</option>
                    <option value="18-25">18-25岁</option>
                    <option value="26-35">26-35岁</option>
                    <option value="36-45">36-45岁</option>
                    <option value="46-55">46-55岁</option>
                    <option value="56+">56岁以上</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="gender">性别:</label>
                <select name="gender" id="gender" required>
                    <option value="">请选择</option>
                    <option value="male">男性</option>
                    <option value="female">女性</option>
                    <option value="other">其他</option>
                    <option value="prefer_not_to_say">不愿透露</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="education">教育程度:</label>
                <select name="education" id="education" required>
                    <option value="">请选择</option>
                    <option value="high_school">高中及以下</option>
                    <option value="bachelor">本科</option>
                    <option value="master">硕士</option>
                    <option value="phd">博士</option>
                    <option value="other">其他</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="experience">AI/ML经验:</label>
                <select name="experience" id="experience" required>
                    <option value="">请选择</option>
                    <option value="none">无经验</option>
                    <option value="basic">基础了解</option>
                    <option value="intermediate">有一定经验</option>
                    <option value="advanced">经验丰富</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="video_consumption">日均观看视频时长:</label>
                <select name="video_consumption" id="video_consumption" required>
                    <option value="">请选择</option>
                    <option value="less_1h">少于1小时</option>
                    <option value="1-3h">1-3小时</option>
                    <option value="3-5h">3-5小时</option>
                    <option value="more_5h">5小时以上</option>
                </select>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <button type="submit" class="btn">继续</button>
            </div>
        </form>
    </div>
</body>
</html>
    """
    
    # 实验页面模板
    experiment_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频评价实验</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .header { text-align: center; margin-bottom: 30px; }
        .progress-bar { width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; margin: 20px 0; }
        .progress-fill { height: 100%; background: #28a745; border-radius: 10px; transition: width 0.3s; }
        .video-section { margin: 20px 0; }
        .video-container { display: flex; gap: 20px; margin: 20px 0; }
        .video-player { flex: 1; text-align: center; }
        .video-player video { width: 100%; max-width: 500px; border: 2px solid #ddd; border-radius: 8px; }
        .evaluation-form { margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 8px; }
        .scale-container { margin: 15px 0; }
        .scale-labels { display: flex; justify-content: space-between; margin: 10px 0; font-size: 14px; }
        .radio-group { display: flex; justify-content: space-between; margin: 10px 0; }
        .radio-group label { display: flex; align-items: center; gap: 5px; }
        .submit-btn { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .submit-btn:hover { background: #0056b3; }
        .instructions { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .timer { text-align: center; font-size: 18px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>视频质量评价</h1>
            <p>第 {{ trial_number }} 个 / 共 {{ total_trials }} 个</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ (trial_number / total_trials * 100) }}%"></div>
            </div>
        </div>
        
        <div class="instructions">
            <h3>评价说明：</h3>
            <ul>
                <li>请仔细观看视频A和视频B</li>
                <li>根据您的真实感受对每个视频进行评分</li>
                <li>最后选择您更偏好的视频</li>
            </ul>
        </div>
        
        <div class="video-section">
            <div class="video-container">
                <div class="video-player">
                    <h3>视频 A</h3>
                    <video controls id="videoA">
                        <source src="{{ trial.video_pair.original_video_path }}" type="video/mp4">
                    </video>
                </div>
                <div class="video-player">
                    <h3>视频 B</h3>
                    <video controls id="videoB">
                        <source src="{{ trial.video_pair.improved_video_path }}" type="video/mp4">
                    </video>
                </div>
            </div>
        </div>
        
        <form id="evaluationForm">
            <div class="evaluation-form">
                <h3>视频A评价</h3>
                
                <div class="scale-container">
                    <label><strong>唇音同步质量</strong></label>
                    <p>视频中嘴唇动作与语音的同步程度</p>
                    <div class="radio-group">
                        <label><input type="radio" name="lip_sync_A" value="1" required> 1分</label>
                        <label><input type="radio" name="lip_sync_A" value="2"> 2分</label>
                        <label><input type="radio" name="lip_sync_A" value="3"> 3分</label>
                        <label><input type="radio" name="lip_sync_A" value="4"> 4分</label>
                        <label><input type="radio" name="lip_sync_A" value="5"> 5分</label>
                    </div>
                    <div class="scale-labels">
                        <span>完全不同步</span>
                        <span>基本同步</span>
                        <span>完美同步</span>
                    </div>
                </div>
                
                <div class="scale-container">
                    <label><strong>表情自然度</strong></label>
                    <p>面部表情的自然和真实程度</p>
                    <div class="radio-group">
                        <label><input type="radio" name="expression_A" value="1" required> 1分</label>
                        <label><input type="radio" name="expression_A" value="2"> 2分</label>
                        <label><input type="radio" name="expression_A" value="3"> 3分</label>
                        <label><input type="radio" name="expression_A" value="4"> 4分</label>
                        <label><input type="radio" name="expression_A" value="5"> 5分</label>
                    </div>
                    <div class="scale-labels">
                        <span>非常不自然</span>
                        <span>比较自然</span>
                        <span>非常自然</span>
                    </div>
                </div>
                
                <div class="scale-container">
                    <label><strong>音频质量</strong></label>
                    <p>音频的清晰度和质量</p>
                    <div class="radio-group">
                        <label><input type="radio" name="audio_quality_A" value="1" required> 1分</label>
                        <label><input type="radio" name="audio_quality_A" value="2"> 2分</label>
                        <label><input type="radio" name="audio_quality_A" value="3"> 3分</label>
                        <label><input type="radio" name="audio_quality_A" value="4"> 4分</label>
                        <label><input type="radio" name="audio_quality_A" value="5"> 5分</label>
                    </div>
                    <div class="scale-labels">
                        <span>质量很差</span>
                        <span>质量一般</span>
                        <span>质量很好</span>
                    </div>
                </div>
                
                <div class="scale-container">
                    <label><strong>视觉清晰度</strong></label>
                    <p>视频的清晰度和视觉质量</p>
                    <div class="radio-group">
                        <label><input type="radio" name="visual_clarity_A" value="1" required> 1分</label>
                        <label><input type="radio" name="visual_clarity_A" value="2"> 2分</label>
                        <label><input type="radio" name="visual_clarity_A" value="3"> 3分</label>
                        <label><input type="radio" name="visual_clarity_A" value="4"> 4分</label>
                        <label><input type="radio" name="visual_clarity_A" value="5"> 5分</label>
                    </div>
                    <div class="scale-labels">
                        <span>模糊不清</span>
                        <span>基本清晰</span>
                        <span>非常清晰</span>
                    </div>
                </div>
                
                <div class="scale-container">
                    <label><strong>整体质量</strong></label>
                    <p>对视频的整体印象</p>
                    <div class="radio-group">
                        <label><input type="radio" name="overall_quality_A" value="1" required> 1分</label>
                        <label><input type="radio" name="overall_quality_A" value="2"> 2分</label>
                        <label><input type="radio" name="overall_quality_A" value="3"> 3分</label>
                        <label><input type="radio" name="overall_quality_A" value="4"> 4分</label>
                        <label><input type="radio" name="overall_quality_A" value="5"> 5分</label>
                    </div>
                    <div class="scale-labels">
                        <span>质量很差</span>
                        <span>质量一般</span>
                        <span>质量很好</span>
                    </div>
                </div>
            </div>
            
            <div class="evaluation-form">
                <h3>视频B评价</h3>
                
                <div class="scale-container">
                    <label><strong>唇音同步质量</strong></label>
                    <div class="radio-group">
                        <label><input type="radio" name="lip_sync_B" value="1" required> 1分</label>
                        <label><input type="radio" name="lip_sync_B" value="2"> 2分</label>
                        <label><input type="radio" name="lip_sync_B" value="3"> 3分</label>
                        <label><input type="radio" name="lip_sync_B" value="4"> 4分</label>
                        <label><input type="radio" name="lip_sync_B" value="5"> 5分</label>
                    </div>
                </div>
                
                <div class="scale-container">
                    <label><strong>表情自然度</strong></label>
                    <div class="radio-group">
                        <label><input type="radio" name="expression_B" value="1" required> 1分</label>
                        <label><input type="radio" name="expression_B" value="2"> 2分</label>
                        <label><input type="radio" name="expression_B" value="3"> 3分</label>
                        <label><input type="radio" name="expression_B" value="4"> 4分</label>
                        <label><input type="radio" name="expression_B" value="5"> 5分</label>
                    </div>
                </div>
                
                <div class="scale-container">
                    <label><strong>音频质量</strong></label>
                    <div class="radio-group">
                        <label><input type="radio" name="audio_quality_B" value="1" required> 1分</label>
                        <label><input type="radio" name="audio_quality_B" value="2"> 2分</label>
                        <label><input type="radio" name="audio_quality_B" value="3"> 3分</label>
                        <label><input type="radio" name="audio_quality_B" value="4"> 4分</label>
                        <label><input type="radio" name="audio_quality_B" value="5"> 5分</label>
                    </div>
                </div>
                
                <div class="scale-container">
                    <label><strong>视觉清晰度</strong></label>
                    <div class="radio-group">
                        <label><input type="radio" name="visual_clarity_B" value="1" required> 1分</label>
                        <label><input type="radio" name="visual_clarity_B" value="2"> 2分</label>
                        <label><input type="radio" name="visual_clarity_B" value="3"> 3分</label>
                        <label><input type="radio" name="visual_clarity_B" value="4"> 4分</label>
                        <label><input type="radio" name="visual_clarity_B" value="5"> 5分</label>
                    </div>
                </div>
                
                <div class="scale-container">
                    <label><strong>整体质量</strong></label>
                    <div class="radio-group">
                        <label><input type="radio" name="overall_quality_B" value="1" required> 1分</label>
                        <label><input type="radio" name="overall_quality_B" value="2"> 2分</label>
                        <label><input type="radio" name="overall_quality_B" value="3"> 3分</label>
                        <label><input type="radio" name="overall_quality_B" value="4"> 4分</label>
                        <label><input type="radio" name="overall_quality_B" value="5"> 5分</label>
                    </div>
                </div>
            </div>
            
            <div class="evaluation-form">
                <h3>偏好比较</h3>
                <p>在两个视频中，您更偏好哪一个？</p>
                <div class="radio-group">
                    <label><input type="radio" name="preference" value="A" required> 更偏好视频A</label>
                    <label><input type="radio" name="preference" value="B"> 更偏好视频B</label>
                    <label><input type="radio" name="preference" value="equal"> 两者无明显差异</label>
                </div>
                
                <div class="scale-container">
                    <label><strong>评论 (可选)</strong></label>
                    <textarea name="comments" rows="4" style="width: 100%; margin-top: 10px;" 
                             placeholder="请描述您选择的原因或任何其他观察..."></textarea>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <button type="submit" class="submit-btn">提交评价</button>
            </div>
        </form>
    </div>
    
    <script>
        document.getElementById('evaluationForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // 收集表单数据
            const formData = new FormData(this);
            const responses = {};
            for (let [key, value] of formData.entries()) {
                responses[key] = value;
            }
            
            // 发送数据
            fetch('/api/submit_evaluation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    participant_id: '{{ participant_id }}',
                    trial_data: {{ trial|tojson }},
                    responses: responses
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('评价提交成功！');
                    window.location.href = '/experiment/{{ participant_id }}';
                } else {
                    alert('提交失败：' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('网络错误，请重试');
            });
        });
    </script>
</body>
</html>
    """
    
    # 完成页面模板
    completion_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实验完成</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); min-height: 100vh; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; text-align: center; }
        h1 { color: #28a745; margin-bottom: 30px; }
        .success-icon { font-size: 60px; color: #28a745; margin-bottom: 20px; }
        .stats { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .btn { display: inline-block; padding: 12px 30px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px; }
        .btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="success-icon">✅</div>
        <h1>实验完成！</h1>
        
        <p>感谢您参与我们的实验！您的反馈对我们非常宝贵。</p>
        
        <div class="stats">
            <h3>您的参与统计</h3>
            <p>实验时长: {{ "%.1f"|format(duration) }} 分钟</p>
            <p>完成评价: {{ completed_trials }} 个</p>
        </div>
        
        <div class="stats">
            <h3>研究的意义</h3>
            <p>您的参与将帮助我们：</p>
            <ul style="text-align: left;">
                <li>改进AI生成视频的质量</li>
                <li>提升用户体验</li>
                <li>推动技术发展</li>
            </ul>
        </div>
        
        <p>如有任何问题，请联系我们：research@example.com</p>
        
        <a href="/" class="btn">返回首页</a>
    </div>
</body>
</html>
    """
    
    # 管理员页面模板
    admin_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实验管理</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        h1 { color: #333; text-align: center; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid #007bff; }
        .stat-number { font-size: 2em; font-weight: bold; color: #007bff; }
        .actions { margin: 30px 0; text-align: center; }
        .btn { display: inline-block; padding: 12px 30px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 10px; }
        .btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>实验管理后台</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ stats.total_participants }}</div>
                <div>总参与人数</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.completed_participants }}</div>
                <div>完成人数</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ "%.1f"|format(stats.completion_rate) }}%</div>
                <div>完成率</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ "%.1f"|format(stats.average_duration) }}</div>
                <div>平均时长(分钟)</div>
            </div>
        </div>
        
        <div class="actions">
            <a href="/api/export_results" class="btn">导出数据</a>
            <button class="btn" onclick="refreshStats()">刷新统计</button>
        </div>
    </div>
    
    <script>
        function refreshStats() {
            fetch('/api/get_statistics')
                .then(response => response.json())
                .then(data => {
                    location.reload();
                });
        }
        
        // 自动刷新
        setInterval(refreshStats, 30000); // 30秒刷新一次
    </script>
</body>
</html>
    """
    
    # 保存模板
    templates = {
        'welcome.html': welcome_template,
        'consent.html': consent_template,
        'demographics.html': demographics_template,
        'experiment.html': experiment_template,
        'completion.html': completion_template,
        'admin.html': admin_template
    }
    
    for filename, content in templates.items():
        with open(templates_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"✅ HTML模板已创建到: {templates_dir}")

def main():
    """主函数"""
    print("🌐 启动主观评价实验服务器...")
    
    # 创建HTML模板
    create_templates()
    
    # 启动服务器
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()