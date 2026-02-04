1️⃣ 进入工程根目录并初始化

cd /path/to/your/project/yolo
git init

3️⃣ 绑定远端仓库
git remote add origin https://github.com/Bohao0214/yolo.git
git remote -v

4️⃣ 首次推送
git pull --rebase origin main
git push -u origin main

方式 B（已有目录，强制对齐远端）
git init
git remote add origin https://github.com/Bohao0214/yolo.git
git fetch origin
git checkout -B main origin/main

🔁 拉取最新代码（推荐）
git pull --rebase origin main

✍️ 提交并推送
git status
git add <files>
git commit -m "Your message"
git push origin main

只让 Git 走代理（不影响 pip/conda）

git config http.proxy  http://127.0.0.1:7897
git config https.proxy http://127.0.0.1:7897

“只对 GitHub 走代理”，那就把这两条也改成 7897
git config http.https://github.com.proxy  http://127.0.0.1:7897
git config http.https://github.com/.proxy http://127.0.0.1:7897