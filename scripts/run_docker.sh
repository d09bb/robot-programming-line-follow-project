#!/usr/bin/env bash
# Run on the host with access to its Docker daemon and graphical session.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if ! docker info >/dev/null 2>&1; then
  echo 'Docker daemon에 접근할 수 없습니다. Docker 사용 권한이 있는 호스트 터미널에서 실행하세요.' >&2
  exit 1
fi
mode="${1:-}"
case "$mode" in
  ''|--headless|--check-gl|--empty-world) ;;
  *) echo '사용법: bash scripts/run_docker.sh [--headless|--check-gl|--empty-world]' >&2; exit 2 ;;
esac
docker build -t line-follower:jazzy .
capture_dir="${PROJECT1_CAPTURE_DIR:-$project_dir/captures}"
mkdir -p "$capture_dir"
capture_dir="$(cd -- "$capture_dir" && pwd)"
run_id="$(date +%s%N)-$$"
container_name="project1-line-follower-$run_id"
# Keep old host-network containers from sharing camera/velocity/odom topics.
ros_domain=$((50 + $$ % 150))
xhost_granted=false
container_started=false
cleanup() {
  if [[ "$container_started" == true ]]; then
    docker stop --time 3 "$container_name" >/dev/null 2>&1 || true
  fi
  if [[ "$xhost_granted" == true ]]; then
    xhost -si:localuser:root >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
options=(--rm --name "$container_name" --network host
         -e "ROS_DOMAIN_ID=$ros_domain" -e "GZ_PARTITION=project1_$run_id"
         -v "$capture_dir:/captures")
# Background runs and redirected logs have no interactive terminal.
if [[ -t 0 && -t 1 ]]; then
  options+=(-it)
fi
launch_options=(capture_dir:=/captures "straight_speed:=${PROJECT1_STRAIGHT_SPEED:-0.45}"
                "reference_controller:=${PROJECT1_REFERENCE_CONTROLLER:-true}")
if [[ "$mode" == '--headless' ]]; then
  launch_options+=(headless:=true)
else
  : "${DISPLAY:?GUI 실행에는 DISPLAY가 필요합니다. --headless 옵션도 사용할 수 있습니다.}"
  # Mesa's X11 transport also uses shared memory, independently of Qt.
  options+=(--ipc=host -e DISPLAY -e QT_QPA_PLATFORM=xcb
            -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  # Authorize only the container's root user for this run, then revoke access.
  xhost +si:localuser:root
  xhost_granted=true
fi
if [[ "${PROJECT1_RENDERER:-software}" == hardware ]]; then
  if [[ -d /dev/dri ]]; then options+=(--device /dev/dri); fi
else
  # Do not use an unmapped NVIDIA EGL driver from the host.
  options+=(-e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe
            -e __GLX_VENDOR_LIBRARY_NAME=mesa -e LIBGL_DRI3_DISABLE=1
            -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json)
fi
printf 'ROS_DOMAIN_ID=%s, GZ_PARTITION=project1_%s\n' "$ros_domain" "$run_id"
command=(ros2 launch line_follower line_follow.launch.py "${launch_options[@]}" verbosity:=4)
if [[ "$mode" == --check-gl ]]; then
  command=(bash -c 'glxinfo -B && exec glxgears -info')
  echo 'OpenGL 검사: 회전하는 톱니바퀴가 보여야 합니다. 종료: Ctrl+C'
elif [[ "$mode" == --empty-world ]]; then
  command=(gz sim -v 4 -r /ws/install/line_follower/share/line_follower/worlds/gui_probe.sdf)
  echo 'GUI 검사: 회색 바닥과 빨간 상자가 보여야 합니다. 종료: Ctrl+C'
else
  echo 'Gazebo와 센서 브리지를 시작합니다. 추종기는 약 20초 뒤 주행을 시작합니다.'
fi
container_started=true
docker run "${options[@]}" line-follower:jazzy "${command[@]}"
