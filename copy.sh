
# 작업 디렉토리 설정 (필요에 따라 수정)
DIRECTORY="/data/datasets/etri2022_whole"

# 파일 확장자 설정
EXTENSION="mp4"

# 출력을 저장할 디렉토리 설정 (필요에 따라 수정)
OUTPUT_DIRECTORY="${DIRECTORY}/output"
mkdir -p $OUTPUT_DIRECTORY

# 파일 처리
for FILE in "$DIRECTORY"/*."$EXTENSION"; do
    # 파일 이름 추출
    FILENAME=$(basename "$FILE")
    
    # ffmpeg 명령어 실행: 중앙 위에서 840x840 자르고 244x244로 리사이즈
    ffmpeg -i "$FILE" -vf "crop=840:840:540:0,scale=244:244" "${OUTPUT_DIRECTORY}/${FILENAME}"
done

echo "처리가 완료되었습니다!"