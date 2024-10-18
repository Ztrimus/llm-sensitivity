# Setting Up SOL Computer on Your Local PC

1. **Open Your Terminal**: Ensure you're connected to the ASU Wi-Fi network or the VPN.
2. **SSH Into SOL**
    - In the terminal, type:
        ```bash
        ssh <ASURITE>@sol.asu.edu
        ```
        OR
        ```bash
        ssh <ASURITE>@login.sol.rc.asu.edu
        ```
    - Enter your password (usually your ASU login credentials).
3. **Start a VS Code Tunnel**
    - In the SOL terminal session, run:
        ```bash
        vscode -t 1-0 -q public -p general --mem=40G
        ```
        For GPU,
        ```bash
        vscode -t 1-0 -q public -p general -G a100:1 --mem=40G
        ```
    - You'll receive a prompt to grant server access with a message like:  
      _"Please log into https://github.com/login/device and use code AB12-24CD"_.
4. **Authorize Access**: Open the provided link, enter the given code, and log in.
5. **Access the Online VS Code**
    - After granting access, you'll get a URL like:  
      _"Open this link in your browser https://vscode.dev/tunnel/ab089/packages/apps/vscode"_.
        - Your tunnel name will be `ab089` in this case.
6. **Connect Local VS Code to the Tunnel**
    - Open VS Code on your PC.
    - Press `Cmd/Ctrl + Shift + P` and type: `"Remote-Tunnels: Connect to Tunnel"`.
    - Select **GitHub** and then choose your tunnel (e.g., `ab089`).
7. **Open a Folder on SOL**
    - In VS Code, click `File -> Open Folder`.
    - Navigate to either `/home/<username>/` (ideal for script files) or `/scratch/<username>/` (best for large files, models, or datasets).
8. **Start Working**
    - You're now connected and can start using the remote environment!

# SOL Resources

-   SOL Computer UI: https://ood05.sol.rc.asu.edu/pun/sys/dashboard/
-   VSCode setup for SOL: https://asurc.atlassian.net/wiki/spaces/RC/pages/1907818602/VSCode
