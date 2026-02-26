#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import sys
import argparse
import torch
import time
import numpy as np
from datetime import datetime
from pathlib import Path


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging, load_config, save_config, set_seed, get_device, format_time, create_experiment_dir
from models import ImprovedMultiTaskTalkingFaceEvaluator, MultiTaskTalkingFaceEvaluator
try:
    from models.advanced_mtl_model import AdvancedMTLTalkingFaceEvaluator
except Exception:
    AdvancedMTLTalkingFaceEvaluator = None
from data import create_dataloaders_from_pkl
from training import Trainer
from evaluation import Evaluator


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="訓練AI生成說話人臉視頻評價模型")
    
    parser.add_argument("--config_path", type=str, default="config/optimized_config.yaml",
                        help="配置文件路徑")
    parser.add_argument("--dataset_path", type=str, default="datasets/ac.pkl",
                        help="數據集路徑 (.pkl)")
    parser.add_argument("--output_dir", type=str, default="experiments_improved",
                        help="輸出目錄")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="")
    parser.add_argument("--use_improved_model", action="store_true", default=True,
                        help="使用改进版模型 (ImprovedMultiTaskTalkingFaceEvaluator)")
    parser.add_argument("--use_advanced_model", action="store_true", default=False,
                        help="使用高级模型 (AdvancedMTLTalkingFaceEvaluator)，包含更灵活的动态任务权重与一致性损失")
    parser.add_argument("--resume", type=str, default=None,
                        help="")
    parser.add_argument("--use_cuda", action="store_true", default=True,
                        help="")
    parser.add_argument("--seed", type=int, default=42,
                        help="")
    
    return parser.parse_args()


def create_improved_trainer(model, config, train_loader, val_loader, test_loader, device):
    """Create an improved trainer for the model."""
    
    class ImprovedTrainer(Trainer):
        """Improved trainer for multi-task video evaluation."""
        
        def __init__(self, model, config, train_loader, val_loader, test_loader, device=None):
            super().__init__(model, config, train_loader, val_loader, test_loader, device)

            # Setup improvements
            self.setup_improvements()
        
        def setup_improvements(self):
            """Setup improvements for training."""
            import torch.optim as optim
            from torch.optim.lr_scheduler import OneCycleLR

            # AdamW optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['optimizer']['lr'],
                weight_decay=self.config['train']['optimizer']['weight_decay'],
                betas=self.config['train']['optimizer'].get('betas', [0.9, 0.999]),
                eps=self.config['train']['optimizer'].get('eps', 1e-8)
            )

            # OneCycleLR scheduler
            if self.config['train']['scheduler']['type'] == 'onecycle':
                total_steps = len(self.train_loader) * self.config['train']['epochs']
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=self.config['train']['scheduler']['max_lr'],
                    total_steps=total_steps,
                    pct_start=self.config['train']['scheduler']['pct_start'],
                    anneal_strategy=self.config['train']['scheduler']['anneal_strategy'],
                    div_factor=self.config['train']['scheduler']['div_factor'],
                    final_div_factor=self.config['train']['scheduler']['final_div_factor']
                )
            else:
                # 使用基类的创建方法（名称为 _create_scheduler）
                self.scheduler = super()._create_scheduler()

            # Gradient clipping
            self.gradient_clip_val = self.config['train'].get('gradient_clip_val', 1.0)

            # Label smoothing
            self.label_smoothing = self.config['train'].get('label_smoothing', 0.0)

            # Mixed precision
            self.use_amp = self.config['train'].get('mixed_precision', False)
            if self.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
        
        def train_epoch(self, epoch):
            """Train for one epoch."""
            self.model.train()
            total_loss = 0
            task_losses = {
                'lip_sync': 0, 'expression': 0, 'audio_quality': 0, 'cross_modal': 0, 'overall': 0
            }
            
            progress_bar = enumerate(self.train_loader)
            for i, batch in progress_bar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Zero gradients
                self.optimizer.zero_grad()

                # Mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predictions, losses = self.model(
                            visual_features=batch['visual_features'],
                            audio_features=batch['audio_features'],
                            keypoint_features=batch.get('keypoints'),
                            au_features=batch.get('au_features'),
                            targets={k: v for k, v in batch.items() if k.endswith('_score')}
                        )
                        total_batch_loss = sum(losses.values())
                    
                    # � �
                    self.scaler.scale(total_batch_loss).backward()
                    
                    # ���j
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    # ���p
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    predictions, losses = self.model(
                        visual_features=batch['visual_features'],
                        audio_features=batch['audio_features'],
                        keypoint_features=batch.get('keypoints'),
                        au_features=batch.get('au_features'),
                        targets={k: v for k, v in batch.items() if k.endswith('_score')}
                    )
                    total_batch_loss = sum(losses.values())
                    
                    # � �
                    total_batch_loss.backward()
                    
                    # ���j
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    # ���p
                    self.optimizer.step()
                
                # ��f`�
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        self.scheduler.step()
                
                # �U_1
                total_loss += total_batch_loss.item()
                for task, loss in losses.items():
                    if task in task_losses:
                        task_losses[task] += loss.item()
                
                # Spۦ
                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch}, Batch {i+1}/{len(self.train_loader)}, Loss: {total_batch_loss.item():.4f}")
            
            # ��sG_1
            avg_loss = total_loss / len(self.train_loader)
            avg_task_losses = {task: loss / len(self.train_loader) for task, loss in task_losses.items()}
            
            return avg_loss, avg_task_losses
        
        def validate(self, epoch):
            """Validate model on validation set."""
            self.model.eval()
            total_loss = 0
            task_losses = {
                'lip_sync': 0, 'expression': 0, 'audio_quality': 0, 'cross_modal': 0, 'overall': 0
            }
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = self._move_batch_to_device(batch)
                    
                    predictions, losses = self.model(
                        visual_features=batch['visual_features'],
                        audio_features=batch['audio_features'],
                        keypoint_features=batch.get('keypoints'),
                        au_features=batch.get('au_features'),
                        targets={k: v for k, v in batch.items() if k.endswith('_score')}
                    )
                    
                    total_loss += sum(losses.values()).item()
                    
                    for task, loss in losses.items():
                        if task in task_losses:
                            task_losses[task] += loss.item()

                    # Collect predictions and targets
                    all_predictions.append(predictions)
                    all_targets.append({k: v for k, v in batch.items() if k.endswith('_score')})

            # Compute average loss
            avg_loss = total_loss / len(self.val_loader)
            avg_task_losses = {task: loss / len(self.val_loader) for task, loss in task_losses.items()}
            
            return avg_loss, avg_task_losses, all_predictions, all_targets
        
        def _move_batch_to_device(self, batch):
            """Move batch tensors to the specified device."""
            device_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    device_batch[key] = value.to(self.device)
                else:
                    device_batch[key] = value
            return device_batch
    
    return ImprovedTrainer(model, config, train_loader, val_loader, test_loader, device)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(args.output_dir, args.experiment_name)

    # Setup logging
    logger = setup_logging(os.path.join(experiment_dir, "logs"))
    logger.info("Logging setup complete. Starting training of AI-generated talking face video evaluation model.")
    logger.info(f"Config path: {args.config_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Experiment directory: {experiment_dir}")

    # Initialize: Trainer, Evaluator, etc.
    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    # Load config
    config = load_config(args.config_path)

    # Update dataset path
    config['data']['dataset_path'] = args.dataset_path
    if 'train' not in config:
        config['train'] = {}
    config['train']['output_dir'] = experiment_dir

    # Save config
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_save_path)
    logger.info(f"Config saved to: {config_save_path}")

    # Initialize model, data loaders, etc.
    device = get_device(args.use_cuda)
    logger.info(f"Device: {device}")

    # Initialize model
    logger.info("Initializing model...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders_from_pkl(
            dataset_pkl_path=args.dataset_path,
            config=config
        )
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return

    # Setup model
    logger.info("Setup model...")
    try:
        if args.use_advanced_model and AdvancedMTLTalkingFaceEvaluator is not None:
            model = AdvancedMTLTalkingFaceEvaluator(config['model'])
            logger.info("Using advanced model.")
        elif args.use_improved_model and ImprovedMultiTaskTalkingFaceEvaluator is not None:
            model = ImprovedMultiTaskTalkingFaceEvaluator(config['model'])
            logger.info("Using improved model.")
        else:
            model = MultiTaskTalkingFaceEvaluator(config['model'])
            logger.info("Using standard model.")

        model.to(device)

        # Summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # Initialize trainer
    try:
        if args.use_advanced_model and AdvancedMTLTalkingFaceEvaluator is not None:
            trainer = create_improved_trainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device
            )
            logger.info("Using advanced model trainer (improved trainer pipeline).")
        elif args.use_improved_model:
            trainer = create_improved_trainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device
            )
            logger.info("Using improved model.")
        else:
            trainer = Trainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device
            )
            logger.info("Using standard model.")
    except Exception as e:
    # 输出完整堆栈以便诊断类型比较错误
      logger.exception(f"Error initializing trainer: {e}")
      return

    # Checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            start_epoch = trainer.load_checkpoint(args.resume)
            logger.info(f"Loaded checkpoint, {start_epoch} n")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return

    # Start training
    logger.info("Starting training...")
    train_start_time = time.time()
    
    try:
        final_results = trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    
    train_time = time.time() - train_start_time
    logger.info(f"Training time: {format_time(train_time)}")
    if 'best_epoch' in final_results:
        logger.info(f"Best epoch: {final_results['best_epoch']}")

    # Start evaluation
    logger.info("Starting evaluation...")
    eval_start_time = time.time()
    
    try:
        # Best model path
        best_model_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')
        logger.info(f"Best model path: {best_model_path}")

        # Load best model
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            logger.warning(f"Best model not found: {best_model_path}")

        # Start evaluation
        evaluator = Evaluator(
            model=model,
            test_loader=test_loader,
            config=config,
            device=device,
            output_dir=os.path.join(experiment_dir, "results")
        )
        
        # gL�0
        metrics = evaluator.evaluate()
        
        eval_time = time.time() - eval_start_time
        logger.info(f"Evaluation time: {format_time(eval_time)}")

        # Generate report
        report_path = evaluator.generate_report()
        logger.info(f"Report saved to: {report_path}")

        # Summary
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        if 'overall' in metrics:
            print(f";SMSE: {metrics['overall'].get('mse', 'N/A')}")
            print(f";SMAE: {metrics['overall'].get('mae', 'N/A')}")
            print(f";SR2: {metrics['overall'].get('r2', 'N/A')}")
            print(f";SPearson: {metrics['overall'].get('pearson', 'N/A')}")

        print("\nTask Results:")
        for task in ['lip_sync', 'expression', 'audio_quality', 'cross_modal']:
            if task in metrics:
                print(f"- {task}:")
                print(f"  MSE: {metrics[task].get('mse', 'N/A')}")
                print(f"  MAE: {metrics[task].get('mae', 'N/A')}")
                print(f"  R�: {metrics[task].get('r2', 'N/A')}")
                print(f"  Pearson: {metrics[task].get('pearson', 'N/A')}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        return

    # Summary
    total_time = train_time + eval_time
    logger.info(f"Total time: {format_time(total_time)}")
    logger.info("Summary of results:")

    # Task weights
    if hasattr(model, 'get_task_weights'):
        final_weights = model.get_task_weights()
        logger.info("Task weights:")
        for task, weight in final_weights.items():
            logger.info(f"  {task}: {weight:.4f}")
    
    return metrics


if __name__ == "__main__":
    main()