"""
训练监控系统：记录优势分位数、Huber截断比例、PPO-clip比例等关键指标
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

class TrainingMonitor:
    """训练过程监控器，记录关键指标到checkpoints目录"""
    
    def __init__(self, checkpoint_dir: str, log_interval: int = 10):
        """
        Args:
            checkpoint_dir: checkpoints目录路径
            log_interval: 记录间隔（每N步记录一次）
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.step_count = 0
        
        # 创建logs目录
        self.logs_dir = self.checkpoint_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志文件
        self.metrics_file = self.logs_dir / "training_metrics.jsonl"
        self.summary_file = self.logs_dir / "metrics_summary.json"
        
        # 指标累积器
        self.metrics_buffer = defaultdict(list)
        self.current_metrics = {}
        
        # 设置日志
        self.logger = self._setup_logger()
        
        self.logger.info(f"TrainingMonitor initialized, logs dir: {self.logs_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"TrainingMonitor_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not logger.handlers:
            handler = logging.FileHandler(self.logs_dir / "monitor.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def record_advantages_stats(self, advantages: torch.Tensor, method: str = "unknown"):
        """记录优势的分位数统计
        
        Args:
            advantages: 优势tensor
            method: 方法名称 (如 "expectile+huber", "z-score+huber")
        """
        if advantages is None or len(advantages) == 0:
            return
            
        # 转换为numpy进行统计
        adv_np = advantages.detach().cpu().numpy().flatten()
        
        # 计算分位数
        p10 = np.percentile(adv_np, 10)
        p50 = np.percentile(adv_np, 50)  # 中位数
        p90 = np.percentile(adv_np, 90)
        
        # 计算其他统计量
        mean_val = np.mean(adv_np)
        std_val = np.std(adv_np)
        min_val = np.min(adv_np)
        max_val = np.max(adv_np)
        
        # 记录指标
        self.current_metrics.update({
            f"advantages_{method}_p10": float(p10),
            f"advantages_{method}_p50": float(p50),
            f"advantages_{method}_p90": float(p90),
            f"advantages_{method}_mean": float(mean_val),
            f"advantages_{method}_std": float(std_val),
            f"advantages_{method}_min": float(min_val),
            f"advantages_{method}_max": float(max_val),
            f"advantages_{method}_count": len(adv_np)
        })
        
        self.logger.info(
            f"Advantages [{method}] - p10: {p10:.4f}, p50: {p50:.4f}, p90: {p90:.4f}, "
            f"mean: {mean_val:.4f}, std: {std_val:.4f}"
        )
    
    def record_huber_clipping_stats(self, 
                                   original_advantages: torch.Tensor,
                                   clipped_advantages: torch.Tensor,
                                   advantage_delta: float,
                                   method: str = "unknown"):
        """记录Huber截断统计
        
        Args:
            original_advantages: 原始优势
            clipped_advantages: 截断后优势
            advantage_delta: 截断阈值
            method: 方法名称
        """
        if original_advantages is None or clipped_advantages is None:
            return
            
        orig_np = original_advantages.detach().cpu().numpy().flatten()
        clip_np = clipped_advantages.detach().cpu().numpy().flatten()
        
        # 计算被截断的元素
        clipped_mask = np.abs(orig_np - clip_np) > 1e-6
        clipped_count = np.sum(clipped_mask)
        total_count = len(orig_np)
        clipped_ratio = clipped_count / total_count if total_count > 0 else 0.0
        
        # 分别统计上截断和下截断
        upper_clipped = np.sum(orig_np > advantage_delta)
        lower_clipped = np.sum(orig_np < -advantage_delta)
        
        # 记录指标
        self.current_metrics.update({
            f"huber_{method}_clipped_ratio": float(clipped_ratio),
            f"huber_{method}_clipped_count": int(clipped_count),
            f"huber_{method}_total_count": int(total_count),
            f"huber_{method}_upper_clipped": int(upper_clipped),
            f"huber_{method}_lower_clipped": int(lower_clipped),
            f"huber_{method}_advantage_delta": float(advantage_delta)
        })
        
        self.logger.info(
            f"Huber clipping [{method}] - ratio: {clipped_ratio:.4f}, "
            f"upper: {upper_clipped}, lower: {lower_clipped}, total: {total_count}"
        )
    
    def record_ppo_clipping_stats(self, 
                                 ratio: torch.Tensor,
                                 clip_range: float = 0.2,
                                 method: str = "unknown"):
        """记录PPO-clip比例统计
        
        Args:
            ratio: PPO比例 ρ = π_new/π_old
            clip_range: PPO截断范围
            method: 方法名称
        """
        if ratio is None:
            return
            
        ratio_np = ratio.detach().cpu().numpy().flatten()
        
        # 计算被PPO截断的比例
        lower_bound = 1.0 - clip_range
        upper_bound = 1.0 + clip_range
        
        clipped_mask = (ratio_np < lower_bound) | (ratio_np > upper_bound)
        clipped_count = np.sum(clipped_mask)
        total_count = len(ratio_np)
        clipped_ratio = clipped_count / total_count if total_count > 0 else 0.0
        
        # 分别统计上下截断
        upper_clipped = np.sum(ratio_np > upper_bound)
        lower_clipped = np.sum(ratio_np < lower_bound)
        
        # 统计比例分布
        ratio_mean = np.mean(ratio_np)
        ratio_std = np.std(ratio_np)
        
        # 记录指标
        self.current_metrics.update({
            f"ppo_clip_{method}_ratio": float(clipped_ratio),
            f"ppo_clip_{method}_clipped_count": int(clipped_count),
            f"ppo_clip_{method}_total_count": int(total_count),
            f"ppo_clip_{method}_upper_clipped": int(upper_clipped),
            f"ppo_clip_{method}_lower_clipped": int(lower_clipped),
            f"ppo_clip_{method}_ratio_mean": float(ratio_mean),
            f"ppo_clip_{method}_ratio_std": float(ratio_std),
            f"ppo_clip_{method}_clip_range": float(clip_range)
        })
        
        self.logger.info(
            f"PPO clipping [{method}] - ratio: {clipped_ratio:.4f}, "
            f"mean_ratio: {ratio_mean:.4f}, std_ratio: {ratio_std:.4f}"
        )
    
    def record_kl_divergence(self, kl_div: torch.Tensor, method: str = "unknown"):
        """记录KL散度统计
        
        Args:
            kl_div: KL散度tensor
            method: 方法名称
        """
        if kl_div is None:
            return
            
        kl_np = kl_div.detach().cpu().numpy().flatten()
        
        kl_mean = np.mean(kl_np)
        kl_std = np.std(kl_np)
        kl_max = np.max(kl_np)
        kl_min = np.min(kl_np)
        
        # 记录指标
        self.current_metrics.update({
            f"kl_div_{method}_mean": float(kl_mean),
            f"kl_div_{method}_std": float(kl_std),
            f"kl_div_{method}_max": float(kl_max),
            f"kl_div_{method}_min": float(kl_min),
            f"kl_div_{method}_count": len(kl_np)
        })
        
        self.logger.info(
            f"KL divergence [{method}] - mean: {kl_mean:.6f}, "
            f"std: {kl_std:.6f}, max: {kl_max:.6f}"
        )
    
    def step(self, global_step: int, force_log: bool = False):
        """执行一步监控，可能触发日志记录
        
        Args:
            global_step: 全局步数
            force_log: 强制记录日志
        """
        self.step_count += 1
        
        # 添加步数和时间戳
        self.current_metrics.update({
            "global_step": global_step,
            "monitor_step": self.step_count,
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 检查是否需要记录
        if force_log or (self.step_count % self.log_interval == 0):
            self._write_metrics()
            self._update_summary()
            self.current_metrics.clear()
    
    def _write_metrics(self):
        """将当前指标写入JSONL文件"""
        if not self.current_metrics:
            return
            
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                json.dump(self.current_metrics, f, ensure_ascii=False)
                f.write('\n')
                
            self.logger.info(f"Metrics written to {self.metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to write metrics: {e}")
    
    def _update_summary(self):
        """更新汇总统计"""
        if not self.current_metrics:
            return
            
        # 将当前指标添加到缓冲区
        for key, value in self.current_metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_buffer[key].append(value)
        
        # 计算汇总统计
        summary = {
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_steps": self.step_count,
            "metrics_summary": {}
        }
        
        for key, values in self.metrics_buffer.items():
            if len(values) > 0:
                summary["metrics_summary"][key] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "latest": float(values[-1])
                }
        
        # 写入汇总文件
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Summary updated: {self.summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to update summary: {e}")
    
    def finalize(self):
        """完成监控，写入最终统计"""
        if self.current_metrics:
            self._write_metrics()
            self._update_summary()
        
        self.logger.info("TrainingMonitor finalized")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """获取最新的指标"""
        return self.current_metrics.copy()
    
    def get_metrics_history(self, metric_name: str) -> List[float]:
        """获取指定指标的历史数据"""
        return self.metrics_buffer.get(metric_name, [])


# 全局监控器实例
_global_monitor: Optional[TrainingMonitor] = None

def get_global_monitor() -> Optional[TrainingMonitor]:
    """获取全局监控器实例"""
    return _global_monitor

def set_global_monitor(monitor: TrainingMonitor):
    """设置全局监控器实例"""
    global _global_monitor
    _global_monitor = monitor

def create_monitor(checkpoint_dir: str, log_interval: int = 10) -> TrainingMonitor:
    """创建并设置全局监控器"""
    monitor = TrainingMonitor(checkpoint_dir, log_interval)
    set_global_monitor(monitor)
    return monitor