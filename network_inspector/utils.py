import pandas as pd
import torch
from tqdm import tqdm

def test(net, dl, device, save_outputs=False, filename=None):
    if save_outputs:
        df_list = []
    print("Evaluating network ...")
    net.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, targets in tqdm(dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # Forward Pass
            scores = net(data)

            if save_outputs:
                outputs = scores.cpu().numpy()[0]
                for i, out in enumerate(outputs):
                    df_list.append({"output": i, "value": out})
                    
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
            
    accuracy = float(num_correct) / float(num_samples)
    print(
        f"Got {num_correct} / {num_samples} with accuracy {accuracy * 100:.2f}"
    )

    if save_outputs:
        df = pd.DataFrame(df_list)
        print(df)

        df.to_parquet(f"{filename}.parquet")

    
    return accuracy
