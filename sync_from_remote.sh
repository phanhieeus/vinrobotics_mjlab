#!/usr/bin/env bash
# Kéo incremental checkpoint/log từ máy GPU thuê (vast.ai) về máy local.
# Chỉ tải file mới hơn lần sync trước (theo đồng hồ REMOTE để tránh lệch giờ).
# Dùng: bash sync_from_remote.sh
set -uo pipefail

# Máy thuê đổi mỗi lần → cho phép truyền qua biến môi trường, khỏi sửa file:
#   REMOTE_HOST=root@1.2.3.4 REMOTE_PORT=12345 bash sync_from_remote.sh
REMOTE_HOST="${REMOTE_HOST:-root@14.179.88.48}"
REMOTE_PORT="${REMOTE_PORT:-34124}"
SSH="ssh -p $REMOTE_PORT -o BatchMode=yes -o ConnectTimeout=20 $REMOTE_HOST"
REMOTE_DIR="/workspace/vinrobotics_mjlab"
# Giải nén thẳng vào gốc repo: logs/ remote sẽ hoà vào logs/ local, giữ nguyên
# format logs/rsl_rl/<experiment>/<datetime>/ — không tạo thư mục trung gian.
LOCAL_DIR="/d/HocLieu/vinrobotics_mjlab"
STATE="$LOCAL_DIR/logs/.last_sync_epoch"
# Các đường (tương đối trong REMOTE_DIR) cần kéo về:
PATHS="logs wandb train.log"

mkdir -p "$LOCAL_DIR/logs"

LAST=0
[ -f "$STATE" ] && LAST=$(cat "$STATE" 2>/dev/null || echo 0)

# Mốc "bây giờ" theo đồng hồ remote (chốt trước khi tar để lần sau không sót).
NOW=$($SSH "date +%s") || { echo "[sync] SSH lỗi, bỏ qua lần này"; exit 1; }

echo "[sync] $(date '+%F %T') kéo file mới hơn epoch=$LAST ..."

# Đếm số file mới hơn LAST.
TMP_LIST=$($SSH "cd '$REMOTE_DIR' && find $PATHS -type f -newermt '@$LAST' 2>/dev/null | wc -l")
if [ "${TMP_LIST:-0}" -eq 0 ]; then
  echo "[sync] không có file mới."
else
  # Tải về tarball tạm. tar trên remote có thể exit 1 vì file đang được ghi
  # ("file changed as we read it") -> chỉ là cảnh báo, archive vẫn hợp lệ.
  # Nên ta KHÔNG dựa vào exit code của ssh/tar-create, mà kiểm tra archive.
  TARGZ="$LOCAL_DIR/.incoming.$$.tar.gz"
  $SSH "cd '$REMOTE_DIR' && find $PATHS -type f -newermt '@$LAST' -print0 2>/dev/null | tar --null --warning=no-file-changed --ignore-failed-read -czf - --files-from=- 2>/dev/null" > "$TARGZ" || true
  if [ -s "$TARGZ" ] && tar tzf "$TARGZ" >/dev/null 2>&1; then
    tar xzf "$TARGZ" -C "$LOCAL_DIR" \
      && echo "[sync] đã kéo $TMP_LIST file về $LOCAL_DIR" \
      || { echo "[sync] LỖI khi giải nén"; rm -f "$TARGZ"; exit 1; }
    rm -f "$TARGZ"
  else
    echo "[sync] LỖI: tarball rỗng/hỏng, giữ nguyên mốc để thử lại lần sau"; rm -f "$TARGZ"; exit 1
  fi
fi

echo "$NOW" > "$STATE"

# Tóm tắt checkpoint đang có ở local:
CKPTS=$(ls "$LOCAL_DIR"/logs/rsl_rl/*/*/model_*.pt 2>/dev/null | wc -l)
echo "[sync] tổng checkpoint ở local: $CKPTS"
