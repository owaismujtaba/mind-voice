
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box



console = Console()



def styled_print(icon: str, text: str, color: str, bold: bool = True, panel: bool = False):
    """Prints formatted text with an icon, color, and optional bold styling inside a panel."""
    style = f"bold {color}" if bold else color
    message = f"{icon} [{style}]{text}[/{style}]"
    
    if panel:
        console.print(Panel(message, expand=False, border_style=color))
    else:
        console.print(message)

def print_criteria(criteria):
    console = Console()
    
    table = Table(title="Extraction Criteria", box=box.HEAVY, highlight=True)
    table.add_column("Parameter", style="bold cyan", justify="left")
    table.add_column("Value", style="bold yellow", justify="left")
    
    icons = ["🧪", "📏", "🔗", "🎭", "🔬", "📊", "⏱", "⏳"]  # Icons for different parameters
    param_names = [
        "Trial Mode", "Trial Unit","Experiment Mode",
        "Trial Boundary", "Trial Type", "Modality",
        "Tmin", "Tmax"
    ]
    
    for icon, name, value in zip(icons, param_names, criteria):
        table.add_row(f"{icon} {name}", str(value))
    
    console.print(table)