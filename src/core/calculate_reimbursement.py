#!/usr/bin/env python3
"""
Legacy Reimbursement Calculator

This script implements the reverse-engineered reimbursement calculation logic
from the legacy ACME Corp system based on analysis of the public dataset.

Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>

The script takes three parameters:
- days: Number of days for the trip (1-30)
- miles: Total miles traveled during the trip
- receipts: Total dollar amount of submitted receipts

Returns: Single numeric reimbursement amount
"""

import sys
import math


def calculate_reimbursement(days, miles, receipts):
    """
    Main reimbursement calculation function.
    
    Implements the legacy system formula: days * 100 + miles * 0.5 + receipts
    
    Args:
        days: Trip duration in days (1-30)
        miles: Total miles traveled
        receipts: Total receipt amount
        
    Returns:
        Calculated reimbursement amount
    """
    # Input validation
    if days < 1 or days > 30:
        raise ValueError("Trip duration must be between 1 and 30 days")
    if miles < 0:
        raise ValueError("Miles cannot be negative")
    if receipts < 0:
        raise ValueError("Receipts cannot be negative")
    
    # Legacy system formula: days * 100 + miles * 0.5 + receipts
    amount = days * 100 + miles * 0.5 + receipts
    
    # Round to 2 decimal places (currency)
    return round(amount, 2)


def main():
    """Main function for command-line usage."""
    # Simple argument parsing without external dependencies
    if len(sys.argv) < 4:
        print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  python3 calculate_reimbursement.py 3 150 75.50", file=sys.stderr)
        print("  python3 calculate_reimbursement.py 5 400 250.00", file=sys.stderr)
        print("  python3 calculate_reimbursement.py 1 50 25.99", file=sys.stderr)
        sys.exit(1)
    
    # Check for verbose flag
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    # Parse required arguments
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
    except (ValueError, IndexError):
        print("Error: Invalid arguments", file=sys.stderr)
        print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Calculate reimbursement
        amount = calculate_reimbursement(days, miles, receipts)
        
        if verbose:
            print(f"Calculation Details:")
            print(f"  Trip Duration: {days} days")
            print(f"  Miles Traveled: {miles}")
            print(f"  Total Receipts: ${receipts:.2f}")
            print(f"  Formula: {days} * 100 + {miles} * 0.5 + {receipts}")
            print(f"  Calculation: {days * 100} + {miles * 0.5} + {receipts} = {amount:.2f}")
        else:
            # Just output the amount (for script usage)
            print(f"{amount:.2f}")
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Calculation error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 