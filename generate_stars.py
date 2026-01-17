import random

def generate_box_shadow(n):
    shadows = []
    for _ in range(n):
        x = random.randint(0, 2000)
        y = random.randint(0, 2000)
        shadows.append(f"{x}px {y}px #FFF")
    return ", ".join(shadows)

print(".stars {")
print(f"  width: 1px; height: 1px; background: transparent; box-shadow: {generate_box_shadow(700)};")
print("  animation: animStar 50s linear infinite;")
print("}")
print(".stars:after {")
print(f"  content: ' '; position: absolute; top: 2000px; width: 1px; height: 1px; background: transparent; box-shadow: {generate_box_shadow(700)};")
print("}")
