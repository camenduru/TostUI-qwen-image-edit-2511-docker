🐣 Please follow me for new updates: https://x.com/camenduru <br />
🔥 Please join our discord server: https://discord.gg/k5BwmmvJJU <br />
🥳 Please become my sponsor: https://github.com/sponsors/camenduru <br />
🍞 TostUI repo: https://github.com/camenduru/TostUI

#### 🍞 TostUI - Qwen Image Edit 2511 (8bit)

https://github.com/user-attachments/assets/a5c76b73-60a0-4234-a944-ac3242725f9e

1.  **Install Docker**\
    [Download Docker Desktop (Windows AMD64)](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe)
    and run it.

2.  **Update the container (optional)**

    ``` bash
    docker stop tostui-qwen-image-edit-2511; docker rm tostui-qwen-image-edit-2511; docker pull camenduru/tostui-qwen-image-edit-2511
    ```

3.  **Run the container**

    ``` bash
    docker run --gpus all -p 3000:3000 --name tostui-qwen-image-edit-2511 camenduru/tostui-qwen-image-edit-2511
    ```

    *Requires NVIDIA GPU (Min 24GB VRAM)*

4.  **Open app**\
    Go to: http://localhost:3000
