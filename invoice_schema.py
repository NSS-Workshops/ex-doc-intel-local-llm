"""
Pydantic models for invoice data validation
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import date


class Address(BaseModel):
    """Address model"""
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class Seller(BaseModel):
    """Seller information model"""
    name: Optional[str] = None
    address: Optional[Address] = None
    tax_id: Optional[str] = None
    iban: Optional[str] = None


class Client(BaseModel):
    """Client information model"""
    name: Optional[str] = None
    address: Optional[Address] = None
    tax_id: Optional[str] = None


class LineItem(BaseModel):
    """Line item model"""
    number: Optional[int] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_of_measure: Optional[str] = None
    net_unit_price: Optional[float] = None
    net_amount: Optional[float] = None
    vat_percent: Optional[float] = None
    gross_amount: Optional[float] = None


class Summary(BaseModel):
    """Invoice summary model"""
    vat_percent: Optional[float] = None
    net_total: Optional[float] = None
    vat_total: Optional[float] = None
    gross_total: Optional[float] = None


class Invoice(BaseModel):
    """Complete invoice model"""
    invoice_number: Optional[str] = None
    issue_date: Optional[str] = None
    currency: Optional[str] = None
    seller: Optional[Seller] = None
    client: Optional[Client] = None
    items: Optional[List[LineItem]] = Field(default_factory=list)
    summary: Optional[Summary] = None
    
    @field_validator('issue_date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date is in ISO format or None"""
        if v is None or v == "":
            return v
        # Try to parse as ISO date
        try:
            # Check if it's already in YYYY-MM-DD format
            if len(v) == 10 and v[4] == '-' and v[7] == '-':
                date.fromisoformat(v)
                return v
        except (ValueError, AttributeError):
            pass
        return v
