import * as React from "react"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline"
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={`shadcn-btn ${variant} ${className ?? ""}`.trim()}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

// Add more shadcn components as needed
