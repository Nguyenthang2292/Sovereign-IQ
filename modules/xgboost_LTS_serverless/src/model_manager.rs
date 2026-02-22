use crate::error::XGBoostError;
use crate::xgboost_inference::XGBoostModel;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;

pub struct ModelManager {
    cache: RwLock<HashMap<String, Arc<XGBoostModel>>>,
    cache_dir: PathBuf,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            cache_dir: PathBuf::from("/tmp"),
        }
    }

    pub fn with_cache_dir(dir: PathBuf) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            cache_dir: dir,
        }
    }

    pub fn load_into_cache(
        &self,
        key: &str,
        path: &std::path::Path,
    ) -> Result<Arc<XGBoostModel>, XGBoostError> {
        let model = Arc::new(XGBoostModel::from_json_file(path)?);
        let mut cache_write = self.cache.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        cache_write.insert(key.to_string(), Arc::clone(&model));
        Ok(model)
    }

    pub fn get_or_load(
        &self,
        symbol: &str,
        timeframe: &str,
        version: &str,
    ) -> Result<Arc<XGBoostModel>, XGBoostError> {
        let cache_key = format!("{}_{}_{}", symbol, timeframe, version);

        {
            let cache_read = self.cache.read().unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(model) = cache_read.get(&cache_key) {
                return Ok(Arc::clone(model));
            }
        }

        let tmp_path = self.cache_dir.join(&cache_key).with_extension("json");
        if !tmp_path.exists() {
            return Err(XGBoostError::ModelNotFoundError(format!(
                "model file not found in cache: {}",
                tmp_path.display()
            )));
        }

        let model = Arc::new(XGBoostModel::from_json_file(&tmp_path)?);

        let mut cache_write = self.cache.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing_model) = cache_write.get(&cache_key) {
            return Ok(Arc::clone(existing_model));
        }
        cache_write.insert(cache_key, Arc::clone(&model));

        Ok(model)
    }
}
