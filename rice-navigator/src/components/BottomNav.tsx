import { Compass, Bell, Leaf } from "lucide-react";
import { motion } from "framer-motion";

const tabs = [
  { key: "explore", label: "Explore", icon: Compass },
  { key: "alerts", label: "Alerts", icon: Bell },
] as const;

interface Props {
  active: string;
  onChange: (tab: string) => void;
}

const BottomNav = ({ active, onChange }: Props) => (
  <>
    {/* Mobile: bottom navbar */}
    <motion.div
      className="md:hidden fixed bottom-0 left-0 right-0 z-[1000] px-3 pb-3 safe-area-bottom"
      initial={{ y: 80 }}
      animate={{ y: 0 }}
      transition={{ type: "spring", damping: 20, delay: 0.2 }}
    >
      <div className="bg-card rounded-2xl px-2 py-2 flex items-center justify-around shadow-xl border border-border/40">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = active === tab.key;
          return (
            <button
              key={tab.key}
              onClick={() => onChange(tab.key)}
              className="flex flex-col items-center gap-1 py-2 px-5 rounded-xl transition-all duration-200 relative"
            >
              {isActive && (
                <motion.div
                  layoutId="activeTabMobile"
                  className="absolute inset-0 bg-primary/10 rounded-xl"
                  transition={{ type: "spring", damping: 25, stiffness: 300 }}
                />
              )}
              <Icon
                className={`w-5 h-5 relative z-10 transition-colors duration-200 ${
                  isActive ? "text-primary" : "text-muted-foreground"
                }`}
                strokeWidth={isActive ? 2.5 : 2}
              />
              <span
                className={`text-[10px] font-semibold relative z-10 transition-colors duration-200 ${
                  isActive ? "text-primary" : "text-muted-foreground"
                }`}
              >
                {tab.label}
              </span>
            </button>
          );
        })}
      </div>
    </motion.div>

    {/* Desktop: left sidebar */}
    <motion.aside
      className="hidden md:flex fixed top-0 left-0 bottom-0 z-[1000] w-20 flex-col items-center py-5 bg-card border-r border-border/40 shadow-lg"
      initial={{ x: -80 }}
      animate={{ x: 0 }}
      transition={{ type: "spring", damping: 20 }}
    >
      <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center mb-6">
        <Leaf className="w-5 h-5 text-primary" />
      </div>
      <nav className="flex flex-col gap-2 w-full px-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = active === tab.key;
          return (
            <button
              key={tab.key}
              onClick={() => onChange(tab.key)}
              className="flex flex-col items-center gap-1 py-3 rounded-xl transition-all duration-200 relative group"
            >
              {isActive && (
                <motion.div
                  layoutId="activeTabDesktop"
                  className="absolute inset-0 bg-primary/10 rounded-xl"
                  transition={{ type: "spring", damping: 25, stiffness: 300 }}
                />
              )}
              <Icon
                className={`w-5 h-5 relative z-10 transition-colors ${
                  isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                }`}
                strokeWidth={isActive ? 2.5 : 2}
              />
              <span
                className={`text-[10px] font-semibold relative z-10 transition-colors ${
                  isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                }`}
              >
                {tab.label}
              </span>
            </button>
          );
        })}
      </nav>
    </motion.aside>
  </>
);

export default BottomNav;
