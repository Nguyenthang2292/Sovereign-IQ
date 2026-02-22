param(
    [string]$Namespace = "ATC/Serverless",
    [string]$Region = "us-east-1",
    [string]$WarningTopicArn = "",
    [string]$CriticalTopicArn = "",
    [string]$PerformanceTopicArn = "",
    [string]$ErrorTopicArn = ""
)

$ErrorActionPreference = "Stop"

function Set-MetricAlarm {
    param(
        [string]$AlarmName,
        [string]$AlarmDescription,
        [string]$MetricName,
        [string]$Statistic,
        [int]$Period,
        [int]$EvaluationPeriods,
        [double]$Threshold,
        [string]$ComparisonOperator,
        [string]$AlarmActionArn = ""
    )

    $args = @(
        "cloudwatch", "put-metric-alarm",
        "--region", $Region,
        "--alarm-name", $AlarmName,
        "--alarm-description", $AlarmDescription,
        "--metric-name", $MetricName,
        "--namespace", $Namespace,
        "--statistic", $Statistic,
        "--period", "$Period",
        "--evaluation-periods", "$EvaluationPeriods",
        "--threshold", "$Threshold",
        "--comparison-operator", $ComparisonOperator,
        "--treat-missing-data", "notBreaching"
    )

    if ($AlarmActionArn -ne "") {
        $args += @("--alarm-actions", $AlarmActionArn)
    }

    Write-Host "[INFO] Creating/updating alarm: $AlarmName"
    & aws @args
}

Write-Host "[INFO] Configuring CloudWatch alarms in region $Region (namespace: $Namespace)"

Set-MetricAlarm `
    -AlarmName "ATC-Lambda-MemoryWarning" `
    -AlarmDescription "Memory usage exceeds 512MB threshold" `
    -MetricName "MemoryUsageMB" `
    -Statistic "Maximum" `
    -Period 300 `
    -EvaluationPeriods 1 `
    -Threshold 512 `
    -ComparisonOperator "GreaterThanThreshold" `
    -AlarmActionArn $WarningTopicArn

Set-MetricAlarm `
    -AlarmName "ATC-Lambda-MemoryCritical" `
    -AlarmDescription "CRITICAL: Memory usage exceeds 768MB threshold" `
    -MetricName "MemoryUsageMB" `
    -Statistic "Maximum" `
    -Period 60 `
    -EvaluationPeriods 1 `
    -Threshold 768 `
    -ComparisonOperator "GreaterThanThreshold" `
    -AlarmActionArn $CriticalTopicArn

Set-MetricAlarm `
    -AlarmName "ATC-Lambda-LowThroughput" `
    -AlarmDescription "Processing throughput below 1000 symbols/second" `
    -MetricName "SymbolsPerSecond" `
    -Statistic "Average" `
    -Period 300 `
    -EvaluationPeriods 2 `
    -Threshold 1000 `
    -ComparisonOperator "LessThanThreshold" `
    -AlarmActionArn $PerformanceTopicArn

Set-MetricAlarm `
    -AlarmName "ATC-Lambda-HighErrorRate" `
    -AlarmDescription "Error rate exceeds 10% of processed symbols" `
    -MetricName "ErrorRate" `
    -Statistic "Average" `
    -Period 300 `
    -EvaluationPeriods 2 `
    -Threshold 0.1 `
    -ComparisonOperator "GreaterThanThreshold" `
    -AlarmActionArn $ErrorTopicArn

Write-Host "[INFO] CloudWatch alarm setup completed."
