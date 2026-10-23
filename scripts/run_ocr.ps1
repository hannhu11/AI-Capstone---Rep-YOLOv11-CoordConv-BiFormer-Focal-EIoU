[Windows.Globalization.Language,Windows.Foundation.Metadata,ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine,Windows.Foundation.Metadata,ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation.Metadata,ContentType=WindowsRuntime] | Out-Null

function Get-OcrText($imagePath) {
    $file = [Windows.Storage.StorageFile]::GetFileFromPathAsync((Resolve-Path $imagePath).Path).GetAwaiter().GetResult()
    $stream = $file.OpenAsync([Windows.Storage.FileAccessMode]::Read).GetAwaiter().GetResult()
    $decoder = [Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream).GetAwaiter().GetResult()
    $bitmap = $decoder.GetSoftwareBitmapAsync().GetAwaiter().GetResult()
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
    $result = $engine.RecognizeAsync($bitmap).GetAwaiter().GetResult()
    return $result.Text
}

Get-ChildItem -Filter "field*.png" | ForEach-Object {
    Write-Host "=== $($_.Name) ==="
    Get-OcrText $_.FullName
}
