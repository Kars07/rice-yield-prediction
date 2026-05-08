import { useState, useRef, useEffect } from "react";
import { Search, Mic, MicOff, User, Leaf } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { type RiceField } from "@/data/nigeriaRiceData";

const roles = ["Farmer", "Extension Officer", "Government"] as const;

interface Props {
  riceFields: RiceField[];
  onSearch: (query: string) => void;
  onSelectField: (field: RiceField) => void;
}

const SearchHeader = ({ riceFields, onSearch, onSelectField }: Props) => {
  const [query, setQuery] = useState("");
  const [role, setRole] = useState<typeof roles[number]>("Farmer");
  const [showRoles, setShowRoles] = useState(false);
  const [results, setResults] = useState<RiceField[]>([]);
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    const q = query.toLowerCase();
    const matched = riceFields.filter(
      (f) =>
        f.name.toLowerCase().includes(q) ||
        f.state.toLowerCase().includes(q) ||
        f.lga.toLowerCase().includes(q) ||
        f.details.toLowerCase().includes(q)
    );
    setResults(matched.slice(0, 5));
  }, [query]);

  const handleSelect = (field: RiceField) => {
    setQuery(field.name);
    setResults([]);
    onSelectField(field);
  };

  const toggleVoice = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    if (listening && recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-NG";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setQuery(transcript);
      onSearch(transcript);
      setListening(false);
    };

    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  };

  const [isSearchMode, setIsSearchMode] = useState(false);

  return (
    <motion.div
      className="fixed top-0 left-0 right-0 md:left-20 md:right-auto md:w-[40%] z-[1000] px-4 pt-4 pb-1 safe-area-top"
      initial={{ y: -60 }}
      animate={{ y: 0 }}
      transition={{ type: "spring", damping: 20 }}
    >
      <div className="bg-card rounded-full px-4 py-2.5 flex items-center gap-3 shadow-lg border border-border/40">
        {/* Branding / Search toggle */}
        {!isSearchMode ? (
          <>
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <Leaf className="w-4 h-4 text-primary shrink-0" />
              <span className="text-sm font-semibold text-foreground tracking-tight truncate">Rice Monitor Map</span>
            </div>
            <button
              onClick={() => setIsSearchMode(true)}
              className="w-8 h-8 rounded-full bg-muted flex items-center justify-center shrink-0 border border-border/40 hover:bg-muted/80 transition-colors"
            >
              <Search className="w-4 h-4 text-muted-foreground" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleVoice();
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium"
            >
              {listening ? (
                <MicOff className="w-3.5 h-3.5 animate-pulse text-destructive" />
              ) : (
                <Mic className="w-3.5 h-3.5" />
              )}
              <span>{listening ? "Listening..." : "Voice"}</span>
            </button>
          </>
        ) : (
          <>
            <Search className="w-4 h-4 text-muted-foreground shrink-0" />
            <input
              type="text"
              autoFocus
              placeholder="Search states, LGAs, irrigation schemes…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                onSearch(e.target.value);
              }}
              onBlur={() => {
                if (!query) setTimeout(() => setIsSearchMode(false), 200);
              }}
              className="flex-1 bg-transparent outline-none text-sm text-foreground placeholder:text-muted-foreground"
            />
            <button onClick={toggleVoice} className="shrink-0">
              {listening ? (
                <MicOff className="w-4 h-4 text-destructive animate-pulse" />
              ) : (
                <Mic className="w-4 h-4 text-muted-foreground hover:text-primary transition-colors" />
              )}
            </button>
          </>
        )}
       
      </div>

      {/* Search Results Dropdown */}
      <AnimatePresence>
        {results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="mt-2 bg-card rounded-2xl overflow-hidden shadow-xl border border-border/40"
          >
            {results.map((field) => (
              <button
                key={field.id}
                onClick={() => handleSelect(field)}
                className="w-full text-left px-4 py-3 hover:bg-muted/50 transition-colors flex items-center gap-3 border-b border-border/30 last:border-0"
              >
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{
                    backgroundColor:
                      field.type === "healthy"
                        ? "hsl(var(--field-healthy))"
                        : field.type === "stress"
                        ? "hsl(var(--field-stress))"
                        : field.type === "irrigation"
                        ? "hsl(var(--field-irrigation))"
                        : "hsl(var(--field-growth))",
                  }}
                />
                <div>
                  <p className="text-sm font-medium text-foreground">{field.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {field.state} • {field.lga} • {field.hectares} ha
                  </p>
                </div>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default SearchHeader;
