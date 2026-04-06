package translate

import (
	"fmt"
	"sort"
)

var dialects = map[string]Dialect{}

// Register adds a dialect to the global registry. Typically called from init().
func Register(d Dialect) {
	dialects[d.Name()] = d
}

// Get retrieves a registered dialect by name.
func Get(name string) (Dialect, error) {
	d, ok := dialects[name]
	if !ok {
		return nil, fmt.Errorf("unknown dialect %q (registered: %v)", name, registeredNames())
	}
	return d, nil
}

func registeredNames() []string {
	names := make([]string, 0, len(dialects))
	for k := range dialects {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}
