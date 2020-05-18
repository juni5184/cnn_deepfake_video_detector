# Fake_detector  
python Main_thread.py 로 실행  
  
Main_thread.py 파일에서 video_path 경로 바꿔줘야 함 + 해당 파일이 real인지 fake인지 입력(나중에 txt파일 생성하는데 쓰임)  
텍스트 파일 생성할거면 81 - 83번 라인 주석 해제, 저장 경로 설정
-> ======로 표시 해놨음
  
  
영상으로 띄워서 박스 확인하는 부분 - find_fake.py 에서 291 - 294 번 라인 
  
find_fake.py 303 - 305번 라인은 rnn 데이터 만드는 코드이므로 주석 처리 해도됨 / 필요하면 해제
