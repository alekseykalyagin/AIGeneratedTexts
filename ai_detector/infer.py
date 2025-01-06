import hydra
import pandas as pd
import torch
from dataset import get_test_dataloader
from omegaconf import DictConfig
from transformers import AutoTokenizer, BertForSequenceClassification


@hydra.main(config_path="../configs", config_name="config")
def infer(cfg: DictConfig):

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_cfg.model_name, num_labels=1
    )

    state_dict = torch.load(model_cfg.final_model_path)
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    test_loader = get_test_dataloader(
        data_cfg.test_path, tokenizer, batch_size=data_cfg.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, ids = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["id"],
            )
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()

            for i, pred in enumerate(predictions):
                results.append({"id": ids[i], "generated": pred})

    output_csv = data_cfg.predictions_output
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    infer()
