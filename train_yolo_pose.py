from ultralytics.models.yolo.pose import PoseTrainer

trainer = PoseTrainer(
    overrides={
        # Model and data
        'model': 'yolo11n-pose.pt',
        'data': '/root/autodl-tmp/Dataset_YOLO/data_original.yaml',  # Full dataset instead of debug
        
        # Training parameters
        'epochs': 100,  # Increased for full training
        'batch': 120,   # Larger batch size for multi-GPU (16 per GPU across 8 GPUs)
        'imgsz': 640,
        
        # Multi-GPU setup - use all available GPUs (0-11)
        'device': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # All 12 GPUs
        
        # Optimization settings
        'lr0': 0.01,      # Base learning rate
        'lrf': 0.01,      # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Data loading optimization
        'workers': 32,    # Increased workers for multi-GPU
        'cache': False,   # Disable RAM cache due to large dataset
        
        # Motion mask settings
        # 'use_motion_masks': True,
        # 'motion_cache_dir': '/root/autodl-tmp/Dataset_YOLO/motion_cache',  # Use dataset's motion cache
        
        # Training optimization
        'amp': True,      # Automatic Mixed Precision
        'close_mosaic': 10,
        'patience': 50,   # Early stopping patience
        'save_period': 5, # Save checkpoint every 5 epochs
        
        # Project settings
        'project': '/root/autodl-tmp/experiments/original_yolopose',
        'name': 'yolo11_pose_original_multi_gpu',
        'exist_ok': False,
        'verbose': True,
        
        # Validation settings
        'val': True,
        'plots': True,
        'save_json': False,
    }
)

results = trainer.train()