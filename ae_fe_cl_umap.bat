@REM SET MODEL_NAME=ct_residual_mse_2k

@REM uv run ./src/main.py generate_latent_spaces --checkpoint C:\Users\kubas\Downloads\autoencoder-10k-CT-Residual-mse-latent2k-training-mse-final-21999.ckpt --output-dir data/latent_spaces/%MODEL_NAME%
@REM uv run ./src/main.py fit_feature_extractor --input-dir data/latent_spaces/%MODEL_NAME% --output-dir data/models/ae_%MODEL_NAME%
@REM uv run ./src/main.py generate_latent_components --input-dir data/latent_spaces/%MODEL_NAME% --output-dir data/latent_components/%MODEL_NAME% --checkpoint data/models/ae_%MODEL_NAME%/feature_extractor.pkl
@REM uv run ./src/main.py fit_clusterizer --input-dir data/latent_components/%MODEL_NAME% --output-dir data/models/ae_%MODEL_NAME%  --n-clusters 6
@REM uv run ./src/main.py generate_clusters --input-dir data/latent_components/%MODEL_NAME% --output-dir data/clusters/%MODEL_NAME% --checkpoint data/models/ae_%MODEL_NAME%/clusterizer.pkl
@REM uv run ./src/main.py visualize_umap --latent-components-input-dir data/latent_components/%MODEL_NAME% --clusters-input-dir data/clusters/%MODEL_NAME% --output-dir data/plots/%MODEL_NAME%  


@REM SET MODEL_NAME=ct_residual_mse_8k_to_2k
@REM SET CHECKPOINT_PATH=C:\Users\kubas\Downloads\autoencoder-10k-CT-Residual-mse-latent8k-to-latent4k-to-latent2k-final-49999.ckpt

@REM uv run ./src/main.py generate_latent_spaces --checkpoint %CHECKPOINT_PATH% --output-dir data/latent_spaces/%MODEL_NAME%
@REM uv run ./src/main.py fit_feature_extractor --input-dir data/latent_spaces/%MODEL_NAME% --output-dir data/models/ae_%MODEL_NAME%
@REM uv run ./src/main.py generate_latent_components --input-dir data/latent_spaces/%MODEL_NAME% --output-dir data/latent_components/%MODEL_NAME% --checkpoint data/models/ae_%MODEL_NAME%/feature_extractor.pkl
@REM uv run ./src/main.py fit_clusterizer --input-dir data/latent_components/%MODEL_NAME% --output-dir data/models/ae_%MODEL_NAME%  --n-clusters 6
@REM uv run ./src/main.py generate_clusters --input-dir data/latent_components/%MODEL_NAME% --output-dir data/clusters/%MODEL_NAME% --checkpoint data/models/ae_%MODEL_NAME%/clusterizer.pkl
@REM uv run ./src/main.py visualize_umap --latent-components-input-dir data/latent_components/%MODEL_NAME% --clusters-input-dir data/clusters/%MODEL_NAME% --output-dir data/plots/%MODEL_NAME%  

@REM SET MODEL_NAME=ct_residual_mse_8k
SET CHECKPOINT_PATH=".\checkpoints\autoencoder-10k-CT-Residual-mse-latent8k-final-22301.ckpt"

uv run ./src/main.py generate_latent_spaces --checkpoint %CHECKPOINT_PATH% --output-dir data/latent_spaces
uv run ./src/main.py fit_feature_extractor --input-dir data/latent_spaces/ --output-dir data/models
uv run ./src/main.py generate_latent_components --input-dir data/latent_spaces/ --output-dir data/latent_components/ --checkpoint data/models/feature_extractor.pkl
uv run ./src/main.py fit_clusterizer --input-dir data/latent_components/ --output-dir data/models  --n-clusters 9
uv run ./src/main.py generate_clusters --input-dir data/latent_components/ --output-dir data/clusters/ --checkpoint data/models/clusterizer.pkl
uv run ./src/main.py visualize_umap --latent-components-input-dir data/latent_components/ --clusters-input-dir data/clusters/ --output-dir data/plots/  

