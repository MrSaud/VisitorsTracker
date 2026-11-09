; Inno Setup script for VisitorsTracker.
;
;   pyinstaller VisitorsTracker.spec
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\VisitorsTracker.iss
;
; Produces installer\Output\VisitorsTracker-Setup.exe.
;
; Installs per user, not machine-wide: PrivilegesRequired=lowest means no
; administrator prompt and no "ask your IT department" on a locked-down
; machine. The app only ever writes to the user's own profile anyway.

#define AppName       "VisitorsTracker"
#define AppVersion    "1.0.0"
#define AppPublisher  "VisitorsTracker"
#define AppExeName    "VisitorsTracker.exe"

[Setup]
; Never change this GUID: it is how Windows recognises an upgrade rather
; than a second, parallel installation.
AppId={{4C2A7E18-9F3B-4D61-ADE5-6B0C8F2A1D77}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=Output
OutputBaseFilename=VisitorsTracker-Setup
SetupIconFile=..\assets\AppIcon.ico
UninstallDisplayIcon={app}\{#AppExeName}
; The payload is a PyInstaller onefile, already compressed. Squeezing it
; again costs minutes of build time for a few percent.
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Start {#AppName} now"; \
    Flags: nowait postinstall skipifsilent

; Counting history lives in the user's profile, not here, so uninstalling
; does not throw it away. Say so rather than silently keep it.
[UninstallDelete]
Type: files; Name: "{app}\*.log"
