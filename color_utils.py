class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def print_colored(text: str, color: str):
    print(colored(text, color))


def print_section(title: str, symbol: str = "="):
    line = symbol * 80
    print(f"\n{Colors.CYAN}{Colors.BOLD}{line}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{line}{Colors.RESET}\n")


def print_step(step_num: int, total: int, description: str):
    print(f"{Colors.BRIGHT_BLUE}[Step {step_num}/{total}]{Colors.RESET} {Colors.BOLD}{description}{Colors.RESET}")


def print_success(message: str):
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_info(message: str):
    print(f"{Colors.CYAN}ℹ {message}{Colors.RESET}")


def print_llm_response(title: str, response: str, max_length: int = 200):
    print(f"\n{Colors.BRIGHT_MAGENTA}{'─' * 80}{Colors.RESET}")
    print(f"{Colors.BRIGHT_MAGENTA}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BRIGHT_MAGENTA}{'─' * 80}{Colors.RESET}")
    
    if len(response) > max_length:
        truncated = response[:max_length] + "..."
        print(f"{Colors.WHITE}{truncated}{Colors.RESET}")
        print(f"{Colors.DIM}(Total length: {len(response)} chars, showing first {max_length}){Colors.RESET}")
    else:
        print(f"{Colors.WHITE}{response}{Colors.RESET}")
    
    print(f"{Colors.BRIGHT_MAGENTA}{'─' * 80}{Colors.RESET}\n")


def print_verification_result(result: str, details: str = ""):
    if result.lower() == "yes":
        print(f"{Colors.GREEN}{Colors.BOLD}VERIFIED: {result}{Colors.RESET}")
        if details:
            print(f"{Colors.GREEN}  Details: {details}{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}FAILED: {result}{Colors.RESET}")
        if details:
            print(f"{Colors.RED}  Details: {details}{Colors.RESET}")


def print_fact(fact_id: str, fact_data: dict):
    print(f"{Colors.GREEN}{Colors.BOLD}{fact_id}:{Colors.RESET} {Colors.WHITE}{fact_data.get('name', 'Unknown')}{Colors.RESET}")
    if 'bbox' in fact_data:
        bbox = fact_data['bbox']
        print(f"  {Colors.DIM}BBox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]{Colors.RESET}")
    if 'relation' in fact_data and fact_data['relation']:
        rel = fact_data['relation']
        print(f"  {Colors.CYAN}Relation: {rel.get('type', 'unknown')} → {rel.get('target', 'unknown')}{Colors.RESET}")
