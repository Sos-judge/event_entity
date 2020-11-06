## Configuration files
In this system, experiments are configured using JSON files.
This readme explains the main attributes of three JSON configuration files:
* build_features_config.json
* train_config.json
* test_config.json

## Configuration file for feature extraction (build_features_config.json):

Used with the script src/features/build_features.py .
To extract features for event and entity mentions, it requires two types of input files
for each split (train/dev/test):
* A json file contains its mention objects (e.g. `train_event_mentions`).
* text file contains its sentences (e.g. `train_text_file`).

Notes:
* The provided build_features_config.json file is configured to extract joint features for event
and entity mentions (with predicate-argument structures extraction).
* SwiRL system's output on the ECB+ corpus is provided with this repo (its directory should be assigned to the srl_output_path attribute).
* ELMo's files (options_file, weight_file) can be downloaded from - *https://allennlp.org/elmo* (we used Original 5.5B model files).

Most of the attributes are self-explained (e.g. batch_size and lr) , but there are few who need
to be explained:
* `use_dep` - Boole. whether use dependency parse,
* `use_srl` - Boole. whether use srl,
* `use_allen_srl` - Boole. This config is activated when use_srl = True. There are 2 kinds of srl can be
  use, allen SRL(if True) or SwiRL SRL(if False).
* `srl_output_path` - Str. This config is activated when use_srl = True. SRL is before this script,
  and this parameter is the path to the output of SRL step.
* `use_left_right_mentions` - ,
* `relaxed_match_with_gold_mention` - ,
* `load_predicted_mentions` - ,
* `load_elmo` - ,
* `options_file" - ,
* `weight_file` -

## Configuration file for training (train_config.json):

Used with the script src/all_models/train_model.py.
The provided `train_config.json` file is configured to train joint model for cross-document entity and event coreference.

Most of the attributes are self-explained (e.g. batch_size and lr) , but there are few who need
to be explained:
* `char_pretrained_path/char_vocab_path` - initial character embeddings (provided in this repo at data/external/char_embed). 
    The original embeddings are available at *https://github.com/minimaxir/char-embeddings*.
* `char_rep_size` - the character LSTM's hidden size.
* `feature_size` - embedding size of binary features.
* `glove_path` - path to pre-trained word embeddings. We used glove.6B.300d which can be downloaded from *https://nlp.stanford.edu/projects/glove/*.
* `train_path/dev_path` - path to the pickle files of the train/dev sets, created by the build_features script (and can be downloaded from *https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK*).
* `dev_th_range` - threshold range to tune on the validation set.
* `entity_merge_threshold/event_merge_threshold` - merge threshold during training (for entities/events).
* `merge_iters` -  for how many iterations to run the agglomerative clustering step (during both training and testing). We used 2 iterations.
* `patient` - for how many epochs we allow the model continue training without an improvement on the dev set.
* `use_args_feats` - whether to use argument/predicate vectors.
* `use_binary_feats` -  whether to use the coreference binary features.
* `wd_entity_coref_file` - a path to a file (provided in this repo) which contains the predictions of a WD entity coreference system on the ECB+. We used CoreNLP for that purpose.


## Configuration file for testing (test_config.json):

Used with the script src/all_models/predict_model.py .
The provided test_config.json file is configured to test the joint model for cross-document entity and event coreference.

The main attributes of this configuration files are:
* `test_path` - 存放测试数据的路径。path to the pickle file of the test set, created by the build_features script (and can be downloaded from *https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK*).
* `cd_event_model_path` - 事件模型的路径。path to the tested event model file.
* `cd_entity_model_path` - 实体模型的路径。path to the tested entity model file.
* `gpu_num` - -1：表示不想尝试使用cuda；其他值正整数：表示想尝试使用几号gpu。
* `event_merge_threshold/entity_merge_threshold` - merge threshold during testing, tuned on the dev set.
* `use_elmo` - ?
* `use_args_feats`- whether to use argument/predicate vectors.
* `use_binary_feats` -  whether to use the coreference binary features.
* `test_use_gold_mentions` - ?
* `wd_entity_coref_file` - a path to a file (provided) which contains the predictions of a WD entity coreference system on the ECB+. We use CoreNLP for that purpose.
* `wd_entity_coref_file` - a path to a file (provided in this repo) which contains the predictions of a WD entity coreference system on the ECB+. We used CoreNLP for that purpose.
* `merge_iters` - 迭代次数，for i in range(1,config_dict["merge_iters"]+1)
* `load_predicted_topics` - false:使用ecb本来的topic true:使用文档聚类算法预测的topic
* `predicted_topics_path` - 如果上边那个选的true，那么这个就是存储“文档聚类算法预测的topic”的文件的路径。path to a pickle file which contains the predicted topics, provided in this repo at data/external/document_clustering or can be obtained using the code in the folder src/doc_clustering.
* `seed` - torch.manual_seed(config_dict["seed"])和torch.cuda.manual_seed(config_dict["seed"])
* `random_seed` - random.seed(config_dict["random_seed"])和np.random.seed(config_dict["random_seed"])
* `event_gold_file_path` - path to the key (gold) event coreference file (for running the evaluation with the CoNLL scorer), provided in this repo.
* `entity_gold_file_path` - path to the key (gold) entity coreference file (for running the evaluation with the CoNLL scorer), provided in this repo.

