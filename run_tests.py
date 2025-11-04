#!/usr/bin/env python
"""
Script para ejecutar los tests sin usar pytest
Esto evita conflictos con plugins de ROS
"""
import sys
import os
import traceback

# Agregar el directorio raíz al path para que Python encuentre los módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    """Ejecuta todos los tests de manera manual"""
    tests_passed = 0
    tests_failed = 0
    failed_tests = []
    
    print("=" * 60)
    print("Ejecutando tests de preprocess.py")
    print("=" * 60)
    
    # Importar y ejecutar tests de preprocess
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_preprocess", 
            os.path.join(os.path.dirname(__file__), "test", "test_preprocess.py")
        )
        test_preprocess = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_preprocess)
        
        test_to_gray_rgb = test_preprocess.test_to_gray_rgb
        test_resize_shape = test_preprocess.test_resize_shape
        test_resize_chw = test_preprocess.test_resize_chw
        test_normalize_uint8 = test_preprocess.test_normalize_uint8
        test_normalize_float = test_preprocess.test_normalize_float
        
        preprocess_tests = [
            ("test_to_gray_rgb", test_to_gray_rgb),
            ("test_resize_shape", test_resize_shape),
            ("test_resize_chw", test_resize_chw),
            ("test_normalize_uint8", test_normalize_uint8),
            ("test_normalize_float", test_normalize_float),
        ]
        
        for test_name, test_func in preprocess_tests:
            try:
                print(f"\n✓ Ejecutando {test_name}...", end=" ")
                test_func()
                print("PASÓ")
                tests_passed += 1
            except AssertionError as e:
                print(f"FALLÓ")
                print(f"  Error: {e}")
                tests_failed += 1
                failed_tests.append((test_name, str(e)))
            except Exception as e:
                print(f"ERROR")
                print(f"  Excepción: {e}")
                traceback.print_exc()
                tests_failed += 1
                failed_tests.append((test_name, str(e)))
                
    except ImportError as e:
        print(f"Error importando tests de preprocess: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Ejecutando tests de wrappers.py")
    print("=" * 60)
    
    # Importar y ejecutar tests de wrappers
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_wrappers", 
            os.path.join(os.path.dirname(__file__), "test", "test_wrappers.py")
        )
        test_wrappers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_wrappers)
        
        test_action_repeat = test_wrappers.test_action_repeat
        test_preprocess_obs_gray_resize_normalize = test_wrappers.test_preprocess_obs_gray_resize_normalize
        test_framestack_basic = test_wrappers.test_framestack_basic
        test_make_env_chain = test_wrappers.test_make_env_chain
        
        wrapper_tests = [
            ("test_action_repeat", test_action_repeat),
            ("test_preprocess_obs_gray_resize_normalize", test_preprocess_obs_gray_resize_normalize),
            ("test_framestack_basic", test_framestack_basic),
            ("test_make_env_chain", test_make_env_chain),
        ]
        
        for test_name, test_func in wrapper_tests:
            try:
                print(f"\n✓ Ejecutando {test_name}...", end=" ")
                test_func()
                print("PASÓ")
                tests_passed += 1
            except AssertionError as e:
                print(f"FALLÓ")
                print(f"  Error: {e}")
                tests_failed += 1
                failed_tests.append((test_name, str(e)))
            except Exception as e:
                print(f"ERROR")
                print(f"  Excepción: {e}")
                traceback.print_exc()
                tests_failed += 1
                failed_tests.append((test_name, str(e)))
                
    except ImportError as e:
        print(f"Error importando tests de wrappers: {e}")
        traceback.print_exc()
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Tests pasados: {tests_passed}")
    print(f"Tests fallidos: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")
    
    if failed_tests:
        print("\nTests que fallaron:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
    
    return 0 if tests_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(run_tests())

