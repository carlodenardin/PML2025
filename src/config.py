class ExperimentConfig:
    # Dataset
    data_dir: str = "data/dsprites"
    batch_size: int = 512
    num_workers: int = 8
    train_val_test_split: list = [0.7, 0.15, 0.15]

    # Model
    latent_dim: int = 10
    lr_vae: float = 1e-4
    beta: float = 1
    lr_disc: float = 1e-4
    gamma: float = 35.0
    disc_hidden_units: int = 1000
    disc_layers: int = 6

    # Training
    epochs: int = 5
    seed: int = 11
    patience_early_stopping: int = 5
    precision: str = "16-mixed"

    # Visualization
    run_visualizations: bool = True
    n_reconstruction_images: int = 8
    n_images_for_static_traversals: int = 3
    n_traversal_steps_per_dim: int = 11
    traversal_range_min: float = -2.5
    traversal_range_max: float = 2.5

    # Paths
    base_output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"