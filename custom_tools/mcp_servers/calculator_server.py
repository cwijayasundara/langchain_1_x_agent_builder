"""
Calculator MCP Server - Provides mathematical operation tools.

This MCP server exposes calculator tools via HTTP transport.
It can be connected to by LangChain agents through the MCP adapter.

Usage:
    python calculator_server.py

    The server will run on http://localhost:8005/mcp

Configuration in agent YAML:
    mcp_servers:
      - name: calculator
        description: Mathematical operations
        transport: streamable_http
        url: http://localhost:8005/mcp
"""

import math
import uvicorn
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Calculator")


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b

    Examples:
        add(5, 3) -> 8.0
        add(-2, 7) -> 5.0
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    Subtract b from a.

    Args:
        a: Number to subtract from
        b: Number to subtract

    Returns:
        The difference (a - b)

    Examples:
        subtract(10, 3) -> 7.0
        subtract(5, 8) -> -3.0
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b

    Examples:
        multiply(4, 5) -> 20.0
        multiply(-3, 6) -> -18.0
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide a by b.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        The quotient (a / b)

    Raises:
        ValueError: If b is zero

    Examples:
        divide(10, 2) -> 5.0
        divide(7, 2) -> 3.5
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """
    Raise base to the power of exponent.

    Args:
        base: The base number
        exponent: The exponent

    Returns:
        base raised to the power of exponent

    Examples:
        power(2, 3) -> 8.0
        power(5, 2) -> 25.0
        power(10, -1) -> 0.1
    """
    return math.pow(base, exponent)


@mcp.tool()
def square_root(x: float) -> float:
    """
    Calculate the square root of a number.

    Args:
        x: The number to find the square root of

    Returns:
        The square root of x

    Raises:
        ValueError: If x is negative

    Examples:
        square_root(16) -> 4.0
        square_root(2) -> 1.4142135623730951
    """
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(x)


@mcp.tool()
def modulo(a: float, b: float) -> float:
    """
    Calculate the remainder when a is divided by b.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        The remainder (a mod b)

    Raises:
        ValueError: If b is zero

    Examples:
        modulo(10, 3) -> 1.0
        modulo(17, 5) -> 2.0
    """
    if b == 0:
        raise ValueError("Cannot calculate modulo with zero divisor")
    return a % b


@mcp.tool()
def absolute_value(x: float) -> float:
    """
    Calculate the absolute value of a number.

    Args:
        x: The number

    Returns:
        The absolute value of x

    Examples:
        absolute_value(-5) -> 5.0
        absolute_value(3.7) -> 3.7
    """
    return abs(x)


@mcp.tool()
def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.

    Args:
        n: A non-negative integer

    Returns:
        n! (n factorial)

    Raises:
        ValueError: If n is negative

    Examples:
        factorial(5) -> 120
        factorial(0) -> 1
        factorial(3) -> 6
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return math.factorial(n)


@mcp.tool()
def calculate_percentage(part: float, whole: float) -> float:
    """
    Calculate what percentage 'part' is of 'whole'.

    Args:
        part: The part value
        whole: The whole value

    Returns:
        The percentage

    Raises:
        ValueError: If whole is zero

    Examples:
        calculate_percentage(25, 100) -> 25.0
        calculate_percentage(3, 12) -> 25.0
    """
    if whole == 0:
        raise ValueError("Cannot calculate percentage with zero whole")
    return (part / whole) * 100


if __name__ == "__main__":
    # Run the MCP server using streamable-http transport
    # This starts an independent HTTP server on port 8005
    print("Starting Calculator MCP Server on http://localhost:8005/mcp")
    print("Available tools: add, subtract, multiply, divide, power, square_root, and more")

    # Use uvicorn to run the streamable-http app on the specified port
    uvicorn.run(mcp.streamable_http_app, host="0.0.0.0", port=8005)
