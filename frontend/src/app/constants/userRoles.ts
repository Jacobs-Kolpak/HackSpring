import {
  Award,
  BookOpen,
  Target,
  type LucideIcon,
} from "lucide-react";

export type UserRole = "student" | "researcher" | "government";

export interface UserRoleOption {
  value: UserRole;
  label: string;
  description: string;
  color: string;
  icon: LucideIcon;
  profileShadow: string;
  profileSoftShadow: string;
}

export const DEFAULT_USER_ROLE: UserRole = "researcher";

export const USER_ROLES: UserRoleOption[] = [
  {
    value: "student",
    label: "Студент",
    description: "",
    color: "from-[#14b8a6] to-[#06b6d4]",
    icon: BookOpen,
    profileShadow: "rgba(20, 184, 166, 0.3)",
    profileSoftShadow: "rgba(20, 184, 166, 0.2)",
  },
  {
    value: "researcher",
    label: "Исследователь",
    description: "",
    color: "from-[#38C571] to-[#70D116]",
    icon: Target,
    profileShadow: "rgba(56, 197, 113, 0.3)",
    profileSoftShadow: "rgba(56, 197, 113, 0.2)",
  },
  {
    value: "government",
    label: "Госслужащий",
    description: "",
    color: "from-[#f97316] to-[#fb923c]",
    icon: Award,
    profileShadow: "rgba(249, 115, 22, 0.3)",
    profileSoftShadow: "rgba(249, 115, 22, 0.2)",
  },
];
