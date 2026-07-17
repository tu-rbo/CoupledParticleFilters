#!/usr/bin/env bash
# Usage: scripts/precompute_parallel.sh [cpf-precompute options] -- <category-or-bag> [...]
# Example: scripts/precompute_parallel.sh --models flow hap graspnet -- ikea01 ikea03 laptop03

set -uo pipefail

usage() {
    echo "Usage: $0 [cpf-precompute options] -- <category-or-bag> [...]" >&2
    echo "Pass model/output options before -- and category names or ROS bag paths after it." >&2
}

if command -v cpf-precompute >/dev/null 2>&1; then
    precompute_command=(cpf-precompute)
elif command -v uv >/dev/null 2>&1; then
    precompute_command=(uv run cpf-precompute)
else
    echo "Neither cpf-precompute nor uv is on PATH." >&2
    exit 127
fi

common_args=()
items=()
after_separator=false
for argument in "$@"; do
    if [[ "$argument" == "--" && "$after_separator" == false ]]; then
        after_separator=true
    elif [[ "$after_separator" == true ]]; then
        items+=("$argument")
    else
        common_args+=("$argument")
    fi
done

if [[ "$after_separator" == false || ${#items[@]} -eq 0 ]]; then
    usage
    exit 2
fi

for argument in "${common_args[@]}"; do
    case "$argument" in
        --bagfile|--categories|--all-bags)
            echo "Selection flags belong to the launcher; put work items after --." >&2
            usage
            exit 2
            ;;
    esac
done

log_dir="${PRECOMPUTE_LOG_DIR:-logs/precompute_parallel}"
mkdir -p "$log_dir"

selection_flag=""
for item in "${items[@]}"; do
    if [[ -e "$item" || "$item" == *.bag || "$item" == *.db3 ]]; then
        candidate_selection="--bagfile"
    else
        candidate_selection="--categories"
    fi
    if [[ -n "$selection_flag" && "$selection_flag" != "$candidate_selection" ]]; then
        echo "Do not mix category names and bag paths in one invocation." >&2
        echo "Run separate batches so each GPU starts one cpf-precompute process." >&2
        exit 2
    fi
    selection_flag="$candidate_selection"
done

declare -a gpu0=() gpu1=() gpu2=() gpu3=()
for index in "${!items[@]}"; do
    case $((index % 4)) in
        0) gpu0+=("${items[$index]}") ;;
        1) gpu1+=("${items[$index]}") ;;
        2) gpu2+=("${items[$index]}") ;;
        3) gpu3+=("${items[$index]}") ;;
    esac
done

run_worker() {
    local gpu="$1"
    shift
    local log_file="$log_dir/gpu${gpu}.log"
    local -a selection=("$selection_flag" "$@")

    : > "$log_file"
    printf 'CUDA_VISIBLE_DEVICES=%s cpf-precompute (%s item(s))\n' "$gpu" "$#" >> "$log_file"
    CUDA_VISIBLE_DEVICES="$gpu" "${precompute_command[@]}" "${common_args[@]}" "${selection[@]}" \
        >> "$log_file" 2>&1
}

declare -a pids=()
if [[ ${#gpu0[@]} -gt 0 ]]; then run_worker 0 "${gpu0[@]}" & pids[0]=$!; fi
if [[ ${#gpu1[@]} -gt 0 ]]; then run_worker 1 "${gpu1[@]}" & pids[1]=$!; fi
if [[ ${#gpu2[@]} -gt 0 ]]; then run_worker 2 "${gpu2[@]}" & pids[2]=$!; fi
if [[ ${#gpu3[@]} -gt 0 ]]; then run_worker 3 "${gpu3[@]}" & pids[3]=$!; fi

overall_status=0
for gpu in 0 1 2 3; do
    if [[ -z "${pids[$gpu]:-}" ]]; then
        continue
    fi
    if wait "${pids[$gpu]}"; then
        printf 'GPU %s completed successfully (log: %s/gpu%s.log)\n' "$gpu" "$log_dir" "$gpu"
    else
        printf 'GPU %s failed; see %s/gpu%s.log\n' "$gpu" "$log_dir" "$gpu" >&2
        overall_status=1
    fi
done

exit "$overall_status"
