# Hook Stop de Claude Code : notification Windows sonore quand Claude a fini une tâche.
$ErrorActionPreference = 'Stop'

$titre = 'Claude Code'
$message = "Tâche terminée - $(Split-Path -Leaf (Get-Location))"

try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null

    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml(@"
<toast>
  <visual>
    <binding template="ToastGeneric">
      <text>$([Security.SecurityElement]::Escape($titre))</text>
      <text>$([Security.SecurityElement]::Escape($message))</text>
    </binding>
  </visual>
  <audio src="ms-winsoundevent:Notification.Default"/>
</toast>
"@)

    # AppID de Windows PowerShell, déjà enregistré sur Windows 10/11.
    $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show(
        [Windows.UI.Notifications.ToastNotification]::new($xml))
}
catch {
    # Repli : son système + info-bulle dans la zone de notification.
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    [System.Media.SystemSounds]::Asterisk.Play()
    $icone = New-Object System.Windows.Forms.NotifyIcon
    $icone.Icon = [System.Drawing.SystemIcons]::Information
    $icone.Visible = $true
    $icone.ShowBalloonTip(5000, $titre, $message, [System.Windows.Forms.ToolTipIcon]::Info)
    Start-Sleep -Seconds 5
    $icone.Dispose()
}
