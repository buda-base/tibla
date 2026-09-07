#!/usr/bin/env bash
# Verify the TiBLA v4 commercial-API provenance manifest against local artifacts.
#
# Checks, for images / ground-truth / the three provider detections:
#   * exactly 833 files per manifest, identical page-stem sets across all five;
#   * every per-file SHA-256 matches the committed manifest (relative basenames);
#   * the portable aggregate (sha256 of the bytewise-sorted manifest) matches
#     the value recorded in provenance.json;
# and validates provenance.json syntax.
#
# It does NOT contact any API and needs no network access.
#
# Usage:
#   # self-contained: verify the in-repo prediction tarballs (no network needed)
#   ./verify.sh --detections ./detections
#
#   # or point at already-extracted directories / the gated test set
#   ./verify.sh --images DIR --gt DIR --azure DIR --textract DIR --google DIR
#
# --detections DIR holds the three <id>.labels.tar.gz + SHA256SUMS; the tarballs
# are checksum-verified and extracted, then their labels are checked per-file.
# Any artifact whose DIR is omitted is skipped for the per-file check, but its
# committed manifest is still checked for count, aggregate and stem alignment.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAN="$HERE/sha256"
JSON="$HERE/provenance.json"

IMAGES="" GT="" AZURE="" TEXTRACT="" GOOGLE="" DETECTIONS=""
_WORK=""
cleanup() { [ -n "$_WORK" ] && rm -rf "$_WORK"; }
trap cleanup EXIT
while [ $# -gt 0 ]; do
  case "$1" in
    --images) IMAGES="$2"; shift 2;;
    --gt) GT="$2"; shift 2;;
    --azure) AZURE="$2"; shift 2;;
    --textract) TEXTRACT="$2"; shift 2;;
    --google) GOOGLE="$2"; shift 2;;
    --detections) DETECTIONS="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

fail=0
note() { printf '%s\n' "$*"; }
bad()  { printf 'FAIL: %s\n' "$*"; fail=1; }

expected_agg() { # $1=manifest basename -> aggregate recorded in provenance.json
  python3 - "$JSON" "$1" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); want="sha256/"+sys.argv[2]
for e in j["manifests"]["entries"]:
    if e["file"]==want: print(e["aggregate_sha256"]); break
PY
}

regen() { # $1=srcdir $2=glob -> portable manifest on stdout
  ( cd "$1" && LC_ALL=C ls $2 2>/dev/null | LC_ALL=C sort | while IFS= read -r f; do sha256sum "$f"; done )
}

check_one() { # $1=label $2=manifest-basename $3=srcdir $4=glob
  local label="$1" mname="$2" src="$3" glob="$4" mf="$MAN/$2"
  [ -f "$mf" ] || { bad "$label: committed manifest missing: $mf"; return; }
  local n agg exp
  n=$(wc -l < "$mf"); agg=$(sha256sum "$mf" | cut -d' ' -f1); exp=$(expected_agg "$mname")
  [ "$n" -eq 833 ] || bad "$label: committed manifest has $n lines, expected 833"
  [ "$agg" = "$exp" ] && note "OK    $label: manifest 833 lines, aggregate $agg matches provenance.json" \
                       || bad "$label: aggregate $agg != provenance.json $exp"
  if [ -n "$src" ]; then
    if [ -d "$src" ]; then
      local tmp; tmp="$(mktemp)"; regen "$src" "$glob" > "$tmp"
      if diff -q "$mf" "$tmp" >/dev/null; then note "OK    $label: all 833 per-file checksums match $src"
      else bad "$label: per-file checksums differ from $src"; diff "$mf" "$tmp" | head -6; fi
      rm -f "$tmp"
    else bad "$label: dir not found: $src"; fi
  else note "SKIP  $label per-file check (no dir given; manifest still verified)"; fi
}

note "== provenance.json syntax =="
python3 -c "import json;json.load(open('$JSON'))" 2>/dev/null \
  && note "OK    provenance.json is valid JSON" || bad "provenance.json is not valid JSON"

# If in-repo detection tarballs are provided, checksum-verify and extract them,
# then feed the extracted label dirs into the per-file check below.
if [ -n "$DETECTIONS" ]; then
  note ""; note "== in-repo detection tarballs ($DETECTIONS) =="
  if [ ! -d "$DETECTIONS" ]; then bad "detections dir not found: $DETECTIONS"; else
    if [ -f "$DETECTIONS/SHA256SUMS" ]; then
      if ( cd "$DETECTIONS" && sha256sum -c SHA256SUMS ) >/dev/null 2>&1; then
        note "OK    all tarball SHA-256 match $DETECTIONS/SHA256SUMS"
      else bad "tarball SHA-256 mismatch against $DETECTIONS/SHA256SUMS"; fi
    else bad "missing $DETECTIONS/SHA256SUMS"; fi
    _WORK="$(mktemp -d)"
    for id in azure_di aws_textract google_docai; do
      tb="$DETECTIONS/$id.labels.tar.gz"
      if [ -f "$tb" ]; then
        mkdir -p "$_WORK/$id"; tar -xzf "$tb" -C "$_WORK/$id"
      else bad "missing tarball: $tb"; fi
    done
    # only set if the user did not override explicitly
    [ -z "$AZURE" ]    && AZURE="$_WORK/azure_di/labels"
    [ -z "$TEXTRACT" ] && TEXTRACT="$_WORK/aws_textract/labels"
    [ -z "$GOOGLE" ]   && GOOGLE="$_WORK/google_docai/labels"
  fi
fi

note ""; note "== counts, per-file & aggregate checksums =="
check_one images       images.sha256       "$IMAGES"   '*.jpg'
check_one gt_labels    gt_labels.sha256    "$GT"       '*.txt'
check_one azure_di     azure_di.sha256     "$AZURE"    '*.txt'
check_one aws_textract aws_textract.sha256 "$TEXTRACT" '*.txt'
check_one google_docai google_docai.sha256 "$GOOGLE"   '*.txt'

note ""; note "== stem alignment (from committed manifests) =="
tmpd="$(mktemp -d)"
for m in images gt_labels azure_di aws_textract google_docai; do
  awk '{print $2}' "$MAN/$m.sha256" | sed -E 's/\.(jpg|txt)$//' | LC_ALL=C sort > "$tmpd/$m"
done
align_ok=1
for m in gt_labels azure_di aws_textract google_docai; do
  diff -q "$tmpd/images" "$tmpd/$m" >/dev/null || { bad "stem set images vs $m differs"; align_ok=0; }
done
[ "$align_ok" = 1 ] && note "OK    all five manifests share an identical 833-stem set"
rm -rf "$tmpd"

note ""
[ "$fail" = 0 ] && note "ALL CHECKS PASSED" || note "SOME CHECKS FAILED"
exit "$fail"
