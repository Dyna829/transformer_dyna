from transformer_dyna import Transformer
from transformers import AutoTokenizer
from Dataset import InMemoryDataset
import torch

d_model = 512
d_ff = 2048
head = 8
data = [
    ("Hello world", "你好 世界"),
    ("How are you?", "你 好吗 ？"),
    ("I love coding", "我 爱 编程"),
    ("This is a test", "这是 一个 测试"),
    ("Good morning", "早上 好"),
    ("What's your name?", "你 叫 什么 名字 ？"),
    ("Thank you very much", "非常 感谢"),
    ("See you tomorrow", "明天 见"),
    ("Where is the toilet?", "洗手间 在 哪里 ？"),
    ("I don't understand", "我 不 明白")
]
def collate_fn(batch):
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]
    
    src_enc = tokenizer(src_batch, padding=True, return_tensors="pt")
    tgt_enc = tokenizer(tgt_batch, padding=True, return_tensors="pt")
    
    return {
        "src_input_ids": src_enc["input_ids"],
        "src_attention_mask": src_enc["attention_mask"],
        "tgt_input_ids": tgt_enc["input_ids"][:, :-1],  # 去掉EOS作为输入
        "tgt_labels": tgt_enc["input_ids"][:, 1:],      # 去掉SOS作为标签
    }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased",
                                              cache_dir="./models",
    mirror="https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = InMemoryDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = Transformer(512, 2048, 8, tokenizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    batch = next(iter(dataloader))  # 只取一个batch验证

    # 训练步骤
    model.train()
    logits = model(
        batch["src_input_ids"],
        batch["tgt_input_ids"]
    )
    loss = loss_fn(logits.reshape(-1, len(tokenizer)), batch["tgt_labels"].reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ========== 7. 验证输出 ==========
    print(f"Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        sample_idx = 0
        src_text = data[sample_idx][0]
        pred = torch.argmax(logits[sample_idx], dim=-1)
        decoded = tokenizer.decode(pred, skip_special_tokens=True)
        
        print(f"\n示例验证:")
        print(f"输入: {src_text}")
        print(f"预测: {decoded}")
        print(f"真实: {data[sample_idx][1]}")