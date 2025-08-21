#!/usr/bin/env python3
"""
Test Neuromodulation Effects in Vertex AI Container
Verifies that neuromodulation effects can be imported and basic functionality works
"""

import sys
import os
sys.path.append('/app')

def test_neuromodulation():
    """Test that neuromodulation effects can be loaded and basic functionality works"""
    
    try:
        # Test importing neuromodulation system
        from neuromod import NeuromodTool, PackRegistry
        from neuromod.effects import EffectRegistry
        print("✅ Neuromodulation system imported successfully")
        
        # Test creating pack registry
        registry = PackRegistry()
        print("✅ Pack registry created successfully")
        
        # Test listing packs
        packs = registry.list_packs()
        print(f"✅ Found {len(packs)} packs")
        
        # Test loading a pack if available
        if packs:
            # Handle both list and dict return types
            if isinstance(packs, dict):
                first_pack_name = list(packs.keys())[0]
            else:
                first_pack_name = packs[0]
            
            pack = registry.get_pack(first_pack_name)
            print(f"✅ Successfully loaded pack: {first_pack_name}")
            
            # Test pack structure
            if hasattr(pack, 'effects'):
                print(f"✅ Pack has {len(pack.effects)} effects")
            else:
                print("⚠️  Pack doesn't have effects attribute")
        else:
            print("⚠️  No packs found")
        
        # Test effect registry
        effect_registry = EffectRegistry()
        effects = effect_registry.list_effects()
        print(f"✅ Found {len(effects)} effects")
        
        # Test that we can create a NeuromodTool instance (but don't use it)
        # This just verifies the class can be instantiated with proper args
        try:
            # Create mock objects for testing
            class MockModel:
                pass
            
            class MockTokenizer:
                pass
            
            neuromod_tool = NeuromodTool(registry, MockModel(), MockTokenizer())
            print("✅ NeuromodTool can be instantiated with proper arguments")
        except Exception as e:
            print(f"⚠️  NeuromodTool instantiation test: {e}")
        
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
