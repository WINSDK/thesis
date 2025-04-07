# parameter repo_id: Either a local path of a model: ./mymodel
#                    or a hugging model name: "bert-base-uncased"
def download_model(repo_id, store_path):
    import os
    def get_name(n):
        return os.path.join(n.replace('/', '--'), 'transformers', 'default', '1')
    if os.path.exists(repo_id):
        return repo_id
    model_path = os.path.join(store_path, get_name(repo_id))
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        download_path = snapshot_download(repo_id=repo_id)
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)
        os.symlink(download_path, model_path, target_is_directory=True)
    return model_path

def main():
    download_model("")
    print("Hello from thesis!")


if __name__ == "__main__":
    main()
