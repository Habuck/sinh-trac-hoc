@echo off
rem Tao kernel7.img (Raspberry Pi 3 bare-metal, nhap nhay LED GPIO18) ngay tai thu muc chua file nay
powershell -NoProfile -ExecutionPolicy Bypass -Command "$h='3F04A0E3020680E30110A0E3011CA0E1041080E50110A0E30119A0E11C1080E53F28A0E3012042E2000052E3FCFFFF1A0110A0E30119A0E1281080E53F28A0E3012042E2000052E3FCFFFF1AF0FFFFEA'; $b=New-Object byte[] ($h.Length/2); for($i=0;$i -lt $b.Length;$i++){$b[$i]=[Convert]::ToByte($h.Substring(2*$i,2),16)}; [IO.File]::WriteAllBytes('%~dp0kernel7.img',$b)"
if exist "%~dp0kernel7.img" (
  echo Da tao: %~dp0kernel7.img
) else (
  echo Loi: khong tao duoc kernel7.img
)
pause
