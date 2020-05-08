python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path banana_model/banana_pipeline.config \
    --trained_checkpoint_prefix banana_model/model.ckpt-10000 \
    --output_directory banana_graph