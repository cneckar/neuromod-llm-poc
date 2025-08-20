#!/usr/bin/env python3
"""
Test Neuromodulation Effects in Vertex AI Container
Verifies that neuromodulation effects can be applied to models
"""

import sys
import os
sys.path.append('/app')

def test_neuromodulation():
    """Test that neuromodulation effects can be loaded and applied"""
    
    try:
        # Test importing neuromodulation system
        from neuromod import NeuromodTool
        from neuromod.effects import EffectRegistry
        print("✅ Neuromodulation system imported successfully")
        
        # Test creating neuromodulation tool
        neuromod_tool = NeuromodTool()
        print("✅ Neuromodulation tool created successfully")
        
        # Test listing packs
        packs = neuromod_tool.list_packs()
        print(f"✅ Found {len(packs)} packs")
        
        # Test loading a pack
        if "caffeine" in packs:
            neuromod_tool.load_pack("caffeine")
            print("✅ Caffeine pack loaded successfully")
            
            # Test getting pack config
            config = neuromod_tool.get_pack_config("caffeine")
            print(f"✅ Pack config retrieved: {len(config.get('effects', []))} effects")
        else:
            print("⚠️  Caffeine pack not found")
        
        # Test effect registry
        effect_registry = EffectRegistry()
        effects = effect_registry.list_effects()
        print(f"✅ Found {len(effects)} effects")
        
        print("✅ All neuromodulation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Neuromodulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_neuromodulation()
    sys.exit(0 if success else 1)
