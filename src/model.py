from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str, dropout: float = 0.1, label_smoothing: float = 0.0):
    """
    Create a token classification model with configurable dropout and label smoothing.
    """
    # Load config and modify dropout
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    
    # Set dropout for better generalization
    if hasattr(config, 'dropout'):
        config.dropout = dropout
    if hasattr(config, 'attention_dropout'):
        config.attention_dropout = dropout
    if hasattr(config, 'classifier_dropout'):
        config.classifier_dropout = dropout
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    
    # Store label smoothing for use in loss computation
    model.config.label_smoothing_factor = label_smoothing
    
    return model
