# PVFaultDetection

.\yt-dlp.exe -x --audio-format mp3 --audio-quality 0 `
  --embed-thumbnail --add-metadata `
  "https://www.youtube.com/watch?v=2EPqn-k9Pjk"

  
.\yt-dlp.exe `
  -f bestvideo[ext=mp4]+bestaudio[ext=m4a]/best `
  --merge-output-format mp4 `
  -o "%(title)s.%(ext)s" `
  "https://www.youtube.com/watch?v=2EPqn-k9Pjk"


https://drive.google.com/file/d/1JyNc5kf9AvvO3LAAN6etvW0uXd96UScu/view?usp=drive_link
