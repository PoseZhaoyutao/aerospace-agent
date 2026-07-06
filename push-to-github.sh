#!/usr/bin/env bash
# 一键推送到 GitHub
# 用法:
#   export GITHUB_USER=你的用户名
#   export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   ./push-to-github.sh

set -e

# 检查必需变量
if [ -z "$GITHUB_USER" ]; then
    echo "[错误] 请设置环境变量 GITHUB_USER"
    echo "  export GITHUB_USER=你的GitHub用户名"
    exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "[错误] 请设置环境变量 GITHUB_TOKEN"
    echo "  export GITHUB_TOKEN=ghp_xxxxxxxxxxxx"
    echo ""
    echo "Token 生成方法:"
    echo "  1. 打开 https://github.com/settings/tokens"
    echo "  2. 点击 'Generate new token (classic)'"
    echo "  3. 勾选 'repo' 权限"
    echo "  4. 复制 token"
    exit 1
fi

REPO_NAME="aerospace-agent"
REMOTE_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "=========================================="
echo "  航天导航控制 Agent — GitHub 推送脚本"
echo "=========================================="
echo "GitHub 用户: $GITHUB_USER"
echo "仓库名:      $REPO_NAME"
echo "本地分支:    $(git branch --show-current)"
echo "提交数:      $(git rev-list --count HEAD)"
echo "文件数:      $(git ls-files | wc -l)"
echo "=========================================="

# 检查远程是否已存在
if git remote get-url origin >/dev/null 2>&1; then
    echo "[信息] 远程仓库 origin 已存在，更新 URL..."
    git remote set-url origin "$REMOTE_URL"
else
    echo "[信息] 添加远程仓库 origin..."
    git remote add origin "$REMOTE_URL"
fi

# 推送
echo ""
echo "[推送] 正在推送到 GitHub..."
git push -u origin main
echo ""
echo "=========================================="
echo "  推送成功!"
echo "  仓库地址: https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo "=========================================="
echo ""
echo "Windows 本地克隆命令:"
echo "  git clone https://github.com/${GITHUB_USER}/${REPO_NAME}.git D:\\Project\\aerospace-agent"
