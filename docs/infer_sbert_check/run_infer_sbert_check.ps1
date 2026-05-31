$ApiUrl = if ($env:API_URL) { $env:API_URL } else { "http://127.0.0.1:8000/personality/infer_sbert" }
$Text = if ($env:TEXT) { $env:TEXT } else { "quick test" }

$body = @{ text = $Text } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri $ApiUrl -ContentType "application/json" -Body $body
