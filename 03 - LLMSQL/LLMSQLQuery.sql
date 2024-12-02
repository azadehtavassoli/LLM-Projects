-- Select rows where SubTotal, TaxAmt, or Freight is NULL
-- These NULL values might indicate incomplete or invalid data in the table.
SELECT SalesOrderID, SubTotal, TaxAmt, Freight, TotalDue
FROM Sales.SalesOrderHeader
WHERE SubTotal IS NULL OR TaxAmt IS NULL OR Freight IS NULL;

-- Update rows to set SubTotal to 0 where it is currently NULL
-- This ensures no NULL values exist in the SubTotal column.
UPDATE Sales.SalesOrderHeader
SET SubTotal = 0
WHERE SubTotal IS NULL;

-- Update rows to set TaxAmt to 0 where it is currently NULL
-- This ensures no NULL values exist in the TaxAmt column.
UPDATE Sales.SalesOrderHeader
SET TaxAmt = 0
WHERE TaxAmt IS NULL;

-- Update rows to set Freight to 0 where it is currently NULL
-- This ensures no NULL values exist in the Freight column.
UPDATE Sales.SalesOrderHeader
SET Freight = 0
WHERE Freight IS NULL;

-- Select rows where SubTotal, TaxAmt, or Freight have negative values
-- Negative values might indicate invalid or incorrect data entries.
SELECT SalesOrderID, SubTotal, TaxAmt, Freight, TotalDue
FROM Sales.SalesOrderHeader
WHERE SubTotal < 0 OR TaxAmt < 0 OR Freight < 0;
